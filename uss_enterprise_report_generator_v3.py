import random
import numpy as np
from faker import Faker
from datetime import datetime

fake = Faker()
np.random.seed(datetime.now().microsecond)

# ===== GALACTIC-SCALE CONFIG =====
TECHNOBABBLE_LEXICON = {
    "quantum": ["phase variance", "flux decoherence", "string fragmentation"],
    "subspace": ["domain inversion", "harmonic rift", "polaron bleed"],
    "plasma": ["eddies", "toroidal collapse", "fermion contamination"],
    "bio": ["neural pattern drift", "cellular resequencer decay", "synaptic gap erosion"]
}

SYSTEM_TEMPLATES = {
    # 36 systems with multi-layer dependencies
    "warp_core": {
        "phrases": ["%s in %s manifold", "unstable %s reaction"],
        "subsystems": ["dilithium articulation", "antideuterium flow", "epsilon matrix"],
        "units": ["cochranes", "terawatts", "milliisotons"],
        "fails": ["impulse", "sensors", "artificial_gravity"]
    },
    "transporter_buffer": {
        "phrases": ["%s pattern degradation", "quantum %s in %s alignment"],
        "subsystems": ["Heisenberg matrix", "pattern enhancers", "phase discriminator"],
        "units": ["quantum%", "microrems"],
        "fails": ["replicators", "holodeck_safety"]
    },
    "structural_integrity": {
        "phrases": ["%s stress fractures", "%s field %s"],
        "subsystems": ["duranium grid", "polarized plating", "ablative generators"],
        "units": ["GPa", "newtons/sq.m"],
        "fails": ["inertial_dampers", "atmospheric_containment"]
    },
    # [Add 33 more systems following this pattern...]
}

FAILURE_CHAINS = [
    ("plasma_leak", ["warp_core", "coolant_system", "environmental"]),
    ("subspace_anomaly", ["sensors", "navigation", "shield_modulation"]),
    ("biohazard", ["medical", "quarantine_fields", "decon_protocols"])
]

# ===== MULTIVERSE-SAFE GENERATION =====
def generate_technobabble(system):
    template = SYSTEM_TEMPLATES[system]
    phrase = random.choice(template["phrases"])
    
    # Inject dynamic technobabble
    replacements = []
    for slot in phrase.split("%s"):
        if "quantum" in slot:
            replacements.append(random.choice(TECHNOBABBLE_LEXICON["quantum"]))
        elif "bio" in slot:
            replacements.append(random.choice(TECHNOBABBLE_LEXICON["bio"]))
        else:
            replacements.append(f"{random.choice(template['subsystems'])} "
                              f"({np.random.normal(1.0, 0.3):.2f}{random.choice(template['units'])})")
    
    return phrase % tuple(replacements[:phrase.count("%s")])

def generate_cascade():
    root_failure, systems = random.choice(FAILURE_CHAINS)
    report = []
    
    # Primary failure
    report.append(f"Initial {root_failure} detected in {random.choice(['sector', 'grid', 'deck'])}_{random.randint(1,100)}")
    
    # Cascade effects
    for sys in systems:
        effect = generate_technobabble(sys)
        report.append(f"with secondary {effect.replace('_',' ')}")
    
    return ", ".join(report) + random.choice([".", " (critical)", "! Priority 1"])

# ===== STARFLEET-APPROVED OUTPUT =====
def generate_starfleet_report():
    systems_in_order = []
    note_lines = []
    
    # Generate 2-4 failure chains
    for _ in range(random.randint(2,4)):
        chain = generate_cascade().split(", ")
        note_lines.extend(chain)
        
        # Extract systems from failure chain
        for term in chain:
            for sys in SYSTEM_TEMPLATES:
                if sys.replace("_"," ") in term:
                    systems_in_order.append(sys)
    
    # Remove duplicates while preserving order
    seen = set()
    systems_clean = [x for x in systems_in_order if not (x in seen or seen.add(x))]
    
    return {
        "note": " ".join(note_lines),
        "systems": ";".join(systems_clean)
    }

# ===== EXAMPLE OUTPUT =====
report = generate_starfleet_report()
print(f"""begin note
----------------
USS Enterprise-E Emergency Diagnostic Report
Stardate {random.randint(50000,60000)}.{random.randint(1,9)} | Auth. Code Sigma-{random.randint(1,9)}{chr(65+random.randint(0,25))}

Inspection notes:
{report['note']}
------------------
Affected components:
{report['systems']}
""")