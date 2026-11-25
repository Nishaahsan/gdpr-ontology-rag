"""Ontology layer for the GDPR assistant.

Reads an OWL ontology saved as RDF/XML (the default format of Protege), without any extra library, and answers
"which concepts does this question talk about, and what does the ontology know about them?".

What it does
  1. Loads classes, individuals, object properties and data properties from the .owl file.
  2. Applies two small inference steps: subclass closure (a Controller is an Actor) and inverse properties
     (hasRight <-> rightOf).
  3. Recognises concepts in a question using rdfs:label synonyms and the words inside the concept names.
  4. Collects facts about them and the articles they are defined in, by walking relations such as
     triggeredBy, hasObligation, hasPenaltyTier and definedIn.

Test on its own:   python ontology.py "List all the rights of a data subject"
"""
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Dict, List, Optional

RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XML = "http://www.w3.org/XML/1998/namespace"
OWL = "http://www.w3.org/2002/07/owl#"

MAX_FACTS = 12

# Words that appear in many questions and must not identify a concept on their own
STOP_TOKENS = {"right", "rights", "personal", "processing", "data", "subject", "controller", "processor",
               "protection", "regulation", "information", "notification", "assessment", "records", "designate",
               "obligation", "principle", "penalty", "basis", "interests", "limitation"}

# How a question refers to a whole class ("all the rights", "legal bases", "fines"...)
CLASS_PATTERNS = {
    "Right": {"plural": r"\brights\b", "singular": r"\bright\b"},
    "Obligation": {"plural": r"\b(obligations|duties|responsibilities)\b", "singular": r"\b(obligation|duty|responsibility)\b"},
    "Principle": {"plural": r"\bprinciples\b", "singular": r"\bprinciple\b"},
    "LegalBasis": {"always": r"\b(legal|lawful) bas(?:is|es)\b"},
    "Penalty": {"always": r"\b(fines?|penalt(?:y|ies)|sanctions?)\b"},
}
LIST_WORDS = re.compile(r"\b(all|list|every|each|which|what are)\b")
TYPE_ORDER = ["Right", "Obligation", "Principle", "LegalBasis", "Penalty", "Event"]


# ----------------------------------------------------------------------------- tiny RDF/XML reader
def _local(uri: str) -> str:
    return re.split(r"[#/]", uri)[-1]


def _tag(elem) -> str:
    return elem.tag.split("}")[-1]


def humanize(name: str) -> str:
    s = name.replace("_", " ")
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[a-zA-Z])(?=[0-9])", " ", s)
    return s.lower().strip()


class Ontology:
    def __init__(self):
        self.types = defaultdict(set)                          # subject -> {class names}
        self.subclass = defaultdict(set)                       # class -> {parent classes}
        self.obj = defaultdict(lambda: defaultdict(list))      # subject -> property -> [objects]
        self.data = defaultdict(lambda: defaultdict(list))     # subject -> property -> [literals]
        self._bnodes = count(1)

    # ---- parsing
    @classmethod
    def load(cls, path) -> "Ontology":
        onto = cls()
        root = ET.parse(str(path)).getroot()
        for node in root:
            onto._node(node)
        onto._finish()
        return onto

    def _node(self, elem) -> str:
        about = elem.get(f"{{{RDF}}}about")
        rid = elem.get(f"{{{RDF}}}ID")
        nid = elem.get(f"{{{RDF}}}nodeID")
        subject = _local(about) if about is not None else (rid or f"_:{nid or next(self._bnodes)}")
        if elem.tag != f"{{{RDF}}}Description":
            self.types[subject].add(_tag(elem))
        for child in elem:
            self._prop(subject, child)
        return subject

    def _prop(self, subject, prop):
        pred = _tag(prop)
        res = prop.get(f"{{{RDF}}}resource")
        nid = prop.get(f"{{{RDF}}}nodeID")
        ptype = prop.get(f"{{{RDF}}}parseType")
        if res is not None:
            self._add_obj(subject, pred, _local(res))
        elif nid is not None:
            self._add_obj(subject, pred, f"_:{nid}")
        elif ptype == "Collection":
            for child in prop:
                self._node(child)
        elif len(prop) > 0:
            for child in prop:
                self._add_obj(subject, pred, self._node(child))
        else:
            text = (prop.text or "").strip()
            dt = prop.get(f"{{{RDF}}}datatype") or ""
            if dt.endswith("integer") or dt.endswith("int"):
                try:
                    text = int(text)
                except ValueError:
                    pass
            self.data[subject][pred].append(text)

    def _add_obj(self, s, p, o):
        if p == "type":
            self.types[s].add(o)
        elif p == "subClassOf":
            self.subclass[s].add(o)
        else:
            self.obj[s][p].append(o)

    def _finish(self):
        # inverse properties, e.g. hasRight <-> rightOf
        pairs = [(s, o) for s, props in list(self.obj.items()) for o in props.get("inverseOf", [])]
        for p, q in pairs:
            for s, props in list(self.obj.items()):
                for o in list(props.get(p, [])):
                    if s not in self.obj[o][q]:
                        self.obj[o][q].append(s)
            for s, props in list(self.obj.items()):
                for o in list(props.get(q, [])):
                    if s not in self.obj[o][p]:
                        self.obj[o][p].append(s)
        self.classes = {s for s, t in self.types.items() if "Class" in t and not s.startswith("_:")}
        self.classes |= set(self.subclass)
        self.individuals = sorted(s for s, t in self.types.items()
                                  if not s.startswith("_:") and s not in self.classes and (t & self.classes))

    # ---- reasoning helpers
    def ancestors(self, cls: str) -> set:
        seen, stack = set(), [cls]
        while stack:
            c = stack.pop()
            for p in self.subclass.get(c, ()):
                if p not in seen:
                    seen.add(p); stack.append(p)
        return seen

    def all_types(self, ind: str) -> set:
        out = set(self.types.get(ind, ()))
        for t in list(out):
            out |= self.ancestors(t)
        return out & self.classes

    def members(self, cls: str) -> List[str]:
        return [i for i in self.individuals if cls in self.all_types(i)]

    def article_of(self, ind: str) -> Optional[int]:
        for art in self.obj.get(ind, {}).get("definedIn", []):
            n = self.data.get(art, {}).get("articleNumber")
            if n:
                return int(n[0])
        return None

    def first(self, ind, prop):
        vals = self.data.get(ind, {}).get(prop)
        return vals[0] if vals else None

    # ---- concept recognition
    def keywords(self, ind: str):
        labels = [l.lower() for l in self.data.get(ind, {}).get("label", [])]
        tokens = [t for t in humanize(ind).split() if len(t) >= 5 and t not in STOP_TOKENS and not t.isdigit()]
        return labels, tokens

    def match_individuals(self, q: str) -> List[str]:
        hits = []
        for ind in self.individuals:
            labels, tokens = self.keywords(ind)
            if any(re.search(rf"\b{re.escape(l)}s?\b", q) for l in labels) or \
               any(re.search(rf"\b{re.escape(t)}s?\b", q) for t in tokens):
                hits.append(ind)
        return hits

    def match_classes(self, q: str) -> List[str]:
        listing = bool(LIST_WORDS.search(q))
        out = []
        for cls, pat in CLASS_PATTERNS.items():
            if cls not in self.classes:
                continue
            if "always" in pat and re.search(pat["always"], q):
                out.append(cls)
            elif "plural" in pat and re.search(pat["plural"], q):
                out.append(cls)
            elif "singular" in pat and listing and re.search(pat["singular"], q):
                out.append(cls)
        return out

    # ---- describing facts
    def main_type(self, ind: str) -> str:
        ts = self.all_types(ind)
        return next((t for t in TYPE_ORDER if t in ts), next(iter(sorted(ts)), "Thing"))

    def describe(self, ind: str) -> str:
        t = self.main_type(ind)
        name = humanize(ind)
        desc = self.first(ind, "description")
        parts = [f"{name} ({humanize(t)}): {desc}" if desc else f"{name} is a {humanize(t)}."]
        if t == "Penalty":
            subjects = [s for s, props in self.obj.items() if ind in props.get("hasPenaltyTier", [])]
            if subjects:
                parts.append("Applies to: " + ", ".join(humanize(s) for s in sorted(subjects)[:MAX_FACTS]) + ".")
            return " ".join(parts)
        if (h := self.first(ind, "deadlineHours")) is not None:
            parts.append(f"Deadline: {h} hours.")
        trig = self.obj.get(ind, {}).get("triggeredBy", [])
        if trig:
            parts.append("Triggered by: " + ", ".join(humanize(x) for x in trig) + ".")
        nots = self.obj.get(ind, {}).get("notifies", [])
        if nots:
            parts.append("Notifies: " + ", ".join(humanize(x) for x in nots) + ".")
        for tier in self.obj.get(ind, {}).get("hasPenaltyTier", []):
            eur, pct = self.first(tier, "maxFineEUR"), self.first(tier, "maxFinePercent")
            if eur is not None:
                parts.append(f"Fine tier: {humanize(tier)} (up to {int(eur):,} EUR or {pct} % of turnover).".replace(",", " "))
        return " ".join(parts)

    # ---- the main entry point
    def query(self, question: str) -> "OntologyResult":
        q = question.lower()
        matched = self.match_individuals(q)
        actors = [i for i in matched if "Actor" in self.all_types(i)]
        events = [i for i in matched if "Event" in self.all_types(i)]
        specific = [i for i in matched if i not in actors and i not in events]
        classes = self.match_classes(q)

        chosen: List[str] = []

        def add(ind):
            if ind not in chosen:
                chosen.append(ind)

        for ind in specific:
            add(ind)
        for ev in events:                                   # event -> obligations it triggers
            for ob in self.members("Obligation"):
                if ev in self.obj.get(ob, {}).get("triggeredBy", []):
                    add(ob)
        for cls in classes:                                 # whole-class questions ("all rights")
            pool = self.members(cls)
            if cls == "Obligation":
                if events:
                    pool = [o for o in pool if any(e in self.obj.get(o, {}).get("triggeredBy", []) for e in events)]
                elif actors:
                    mine = {o for a in actors for o in self.obj.get(a, {}).get("hasObligation", [])}
                    pool = [o for o in pool if o in mine] or pool
            for ind in pool:
                add(ind)

        facts, articles, concepts = [], [], []
        for ind in chosen[:MAX_FACTS]:
            art = self.article_of(ind)
            facts.append(Fact(self.describe(ind), art, ind))
            concepts.append(ind)
            if art and art not in articles:
                articles.append(art)
        return OntologyResult(facts, articles, concepts)

    def summary(self) -> str:
        return f"{len(self.classes)} classes, {len(self.individuals)} individuals"


@dataclass
class Fact:
    text: str
    article: Optional[int]
    subject: str

    def cited(self) -> str:
        return f"{self.text} [Art. {self.article}]" if self.article else self.text


@dataclass
class OntologyResult:
    facts: List[Fact] = field(default_factory=list)
    articles: List[int] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)


def find_ontology_file(root: Path) -> Optional[Path]:
    for name in ("gdpr.owl", "gdpr_full.owl"):
        p = root / "ontology" / name
        if p.exists():
            return p
    return None


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    path = find_ontology_file(here)
    if not path:
        sys.exit("No ontology/gdpr.owl found.")
    onto = Ontology.load(path)
    print(f"Loaded {path.name}: {onto.summary()}")
    question = " ".join(sys.argv[1:]) or "List all the rights of a data subject"
    res = onto.query(question)
    print(f"\nQ: {question}\nArticles: {res.articles}\nConcepts: {res.concepts}")
    for f in res.facts:
        print(" -", f.cited())
    if not res.facts:
        print(" (ontology has nothing on this)")
