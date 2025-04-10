"""
USS Enterprise Diagnostic Report Generator
=========================================

Generates realistic system failure reports with:
1. Human-readable symptom descriptions (no system names)
2. Structured system::subsystem mappings
3. Proper Star Trek technobabble

Author: Lt. Commander Data
Stardate: 47988.3
"""
import random
from typing import Dict, List, Tuple, TypedDict, Optional
from dataclasses import dataclass
import numpy as np
from faker import Faker
import textwrap

import re
import json
import requests

class SystemTemplate(TypedDict):
    """Type definition for system configuration templates."""
    phrases: List[str]
    subsystems: List[str]
    units: List[str]
    fails: List[str]


class StarfleetReport(TypedDict):
    """Type definition for generated reports."""
    note: str
    systems: str


@dataclass
class USSEnterpriseDiagnosticGenerator:
    """
    Generates USS Enterprise diagnostic reports.

    Features:
    - 36 fully-defined systems with complete cross-references
    - Cascading failure simulations
    - Training-ready output format
    """

    def __init__(self, refine: bool = False, model_name: str = ''):
        """Initialize with Starfleet-standard configurations."""
        self.refinement_enabled = False
        self.dsClient = None
        self._template = ""
        if refine:
            self.model_name = model_name
            self.refinement_enabled = True
            self.ollama_url = "http://haddock.lab.fluxa.org:11434/api/generate"
            self.ollama_headers = {"Content-Type": "application/json"}
            self._template = """
# Context
A note was generated to mimic technical diagnoses of the USS Enterprise. The note is notorious for having bad quality due
to an evident lack of "human factor", poor semantics and repetition of keywords (like "anomaly", "secondary", etc)

# Instructions
1. You must read the note carefully to identify the main ideas that it contains.
2. Rewrite the note so it contains the exact same main ideas. Be very careful to stick to the requirements.
3. You are allowed to be very slightly creative from time to time.
4. You are encouraged to make the note look like if it was written by a person in a rush.
5. You must answer with the contents of the note, and the contents of the note only.

# Formatting
- The refined note MUST have between 200 and 400 characters.
- The refined note MUST be formatted as a single block of plain-text.

# Variations
- Change the articulation of words in the note to make it "human-like".
- Change the order in which ideas, concepts or facts present themselves in the original note.
- Replace the word "anomaly" by "problem", "issue", "malfunction" or other equivalent expression when needed.

# Note

"""
        self.faker = Faker()
        np.random.seed()

        self.technobabble_lexicon = {
            "quantum": ["phase variance", "flux decoherence", "string fragmentation"],
            "subspace": ["domain inversion", "harmonic rift", "polaron bleed"],
            "plasma": ["eddies", "toroidal collapse", "fermion contamination"],
            "bio": ["neural pattern drift", "cellular resequencing decay", "synaptic gap erosion"]
        }

        self.system_templates = self._initialize_system_templates()
        self.failure_chains = self._initialize_failure_chains()
        self.deck_locations = [f"Deck {i}" for i in range(1, 30)] + [
            "Main Engineering", "Bridge", "Shuttle Bay", "Cargo Bay 3"
        ]

    @property
    def systems(self):
        return self.system_templates


    def _initialize_system_templates(self,
        path: str = "./log_generators/configs/uss_enterprise.json") -> Dict[str, Dict[str, List[str]]]:
        """Initialize all 36 systems with complete configuration and validation."""
        systems = {}
        with open('log_generators/configs/uss_enterprise.json', 'r') as f:
            systems = json.loads(f.read())

        # Final validation
        for sys_name, config in systems.items():
            for failed_system in config["fails"]:
                if failed_system not in systems:
                    raise ValueError(f"System {failed_system} referenced in {sys_name} but not defined")

        return systems

    def _initialize_failure_chains(self) -> List[Tuple[str, List[str]]]:
        """Define failure sequences with validation."""
        chains = [
            ("plasma leak", ["warp core", "coolant system", "environmental"]),
            ("subspace anomaly", ["sensors", "navigation", "shield modulation"]),
            ("biohazard", ["medical", "quarantine fields", "decon protocols"]),
            ("power loss", ["warp core", "impulse drive", "auxiliary power"]),
            ("hull breach", ["structural integrity", "atmospheric containment", "emergency forcefields"]),
            ("weapons malfunction", ["phaser banks", "torpedo launchers", "tactical systems"]),
            ("computer virus", ["computer core", "ops systems", "security protocols"]),
            ("sensor ghosting", ["sensors", "science labs", "tactical systems"])
        ]

        # Validate all systems in chains exist
        for root_cause, systems in chains:
            for sys in systems:
                if sys not in self.system_templates:
                    raise ValueError(f"System {sys} in failure chain with root cause {root_cause} is not defined")

        return chains

    def _get_location(self) -> str:
        """Generate a random ship location."""
        deck_loc = random.choice(self.deck_locations)
        section = random.choice(['A', 'B', 'C', 'D', 'E', 'W', 'X', 'Z'])
        sector = random.choice([1, 2, 3, 5, 8, 13, 61, 99])

        return deck_loc + f", Section {section} / sector {sector}"

    def generate_technobabble(self, system: str) -> str:
        """Generate symptom description without system names."""
        try:
            template = self.system_templates[system]
            phrase = random.choice(template["phrases"])

            symptoms: List[str] = []
            for _ in range(phrase.count("%s")):
                if random.random() > 0.7:
                    category = random.choice(list(self.technobabble_lexicon.keys()))
                    symptoms.append(random.choice(self.technobabble_lexicon[category]))
                else:
                    symptoms.append(f"{random.choice(template['subsystems'])} anomaly")

            return phrase % tuple(symptoms)
        except KeyError:
            return f"unknown system anomaly in {self._get_location()}"

    def generate_failure_chain(self) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Generate a cascading failure sequence."""
        root_cause, systems = random.choice(self.failure_chains)
        symptoms: List[str] = [f"{root_cause.replace('_', ' ')} detected in {self._get_location()}"]
        affected: List[Tuple[str, str]] = []

        for system in systems:
            symptom = self.generate_technobabble(system)
            next_symptom_conn = np.random.choice([
                'with secondary',
                'and',
                ])
            symptoms.append(f"{next_symptom_conn} {symptom}")

            template = self.system_templates.get(system, {})
            subsystem = next(
                (sub for sub in template.get("subsystems", []) if sub in symptom),
                random.choice(template["subsystems"]) if "subsystems" in template else "unknown_subsystem"
            )
            affected.append((system, subsystem))

        return symptoms, affected

    def generate_report(self) -> Dict[str, str]:
        """Generate complete diagnostic report."""
        all_symptoms: List[str] = []
        all_affected: List[Tuple[str, str]] = []

        for _ in range(random.randint(2, 4)):
            symptoms, affected = self.generate_failure_chain()
            all_symptoms.extend(symptoms)
            all_affected.extend(affected)

        # Remove duplicates while preserving order
        seen: Set[Tuple[str, str]] = set()
        unique_affected: List[Tuple[str, str]] = []
        for sys, sub in all_affected:
            if (sys, sub) not in seen:
                seen.add((sys, sub))
                unique_affected.append((sys, sub))

        note = " ".join(all_symptoms) + random.choice([".", "!", ","])
        systems = ";".join(f"{sys}::{sub}" for sys, sub in unique_affected)

        report = {}
        report['note'] = str(note)
        report['systems'] = str(systems)

        return report

    def refine_report(self, report: Dict[str, str]) -> Dict[str, str]:

        if not self.refinement_enabled:
            return report

        content = self._template
        note = report['note']
        content = content + note
        payload = {
            "model": str(self.model_name),
            "prompt": content,
            "stream": False
        }

        response = requests.post(
            self.ollama_url,
            headers=self.ollama_headers,
            data=json.dumps(payload)
        )
        if response.status_code == 200:
            print("[DEBUG] Request was answered by server.")
            response_text = response.text
            payload = json.loads(response_text)
            cot_and_refined_note = payload["response"]
            refined_note = re.sub(r'<think>.*?</think>', '', cot_and_refined_note, flags=re.DOTALL)
            print(f"[DEBUG] Server answered: \n\n{refined_note}\n\n")

            report['note'] = refined_note
        else:
            print(f"[ERROR] {response.status_code} with message {response.text}")
            raise ValueError("Server failure.")

        return report

    def generate_full_report(self, report: Optional[StarfleetReport] = None) -> str:
        """Generate formatted Starfleet report."""
        if report is None:
            report = self.generate_report()

        return f"""
{'-' * 80}\n{str('*' * 22) + '    USS ENTERPRISE-E DIAGNOSTICS    ' + str('*' * 22)}\n{'-' * 80}\n
Stardate {random.randint(50000,60000)}.{random.randint(1,9)} | Auth. Code Sigma-{random.randint(1,9)}{chr(65+random.randint(0,25))}

Inspection notes:
{textwrap.fill(report['note'], 80)}


Affected components:
{textwrap.fill(report['systems'].replace(' ', '_'), 80)}
"""


if __name__ == "__main__":

    import time
    import hashlib
    from tqdm import tqdm
    from pathlib import Path

    model_name = 'deepseek-r1:32b'
    n_reports = 512
    batch = 1

    hash = hashlib.sha1(str(time.asctime()).encode("UTF-8")).hexdigest()
    run_id = hash[0:8]
    print(f"generating entries for run {run_id}")

    generator = USSEnterpriseDiagnosticGenerator(refine=True, model_name=model_name)

    for k in tqdm(range(n_reports)):
        report = generator.generate_report()
        report = generator.refine_report(report)

        Path(f"./txt/{model_name}/{run_id}/").mkdir(parents=True, exist_ok=True)

        # write full report
        full_report = generator.generate_full_report(report=report)
        Path(f"./txt/{model_name}/{run_id}/reports/").mkdir(parents=True, exist_ok=True)
        with open(f"./txt/{model_name}/{run_id}/reports/{k:06d}.txt", 'w') as f:
            f.write(full_report)

        # write note
        Path(f"./txt/{model_name}/{run_id}/notes/").mkdir(parents=True, exist_ok=True)
        with open(f"./txt/{model_name}/{run_id}/notes/{k:06d}.txt", 'w') as f:
            f.write(report['note'])

        # write affected system list
        Path(f"./txt/{model_name}/{run_id}/systems/").mkdir(parents=True, exist_ok=True)
        with open(f"./txt/{model_name}/{run_id}/systems/{k:06d}.txt", 'w') as f:
            f.write(report['systems'])

        k = k + 1
