import random
from faker import Faker
import numpy as np

fake = Faker()
random.seed(42)
np.random.seed(42)

# ===== CONFIGURATION =====
STARBASE_NAMES = ["Deep Space 9", "Starbase 375", "Earth Spacedock", "Starbase 1"]
DECK_GRID = [f"Deck {n}" for n in range(1,82)] + [f"Grid {chr(65+i)}{j}" for i in range(12) for j in range(1,26)]
DEPARTMENTS = ["Engineering", "Science", "Medical", "Security", "Operations"]

SYSTEMS = {
    # Expanded with TNG/DS9/VOY tech
    "warp drive": {
        "issues": ["Plasma conduit resonance at %s", "Dilithium matrix harmonic decay (%s%%)",
                  "Bussard collector efficiency drop", "Warp core breach imminent (Type %s)"],
        "subsystems": ["port nacelle", "starboard nacelle", "injector assembly", "antimatter containment"],
        "severity": lambda: f"E-{random.randint(3,9)}"
    },
    "transporter": {
        "issues": ["Heisenberg variance in pattern buffer %s", "Biofilter degradation (Lifeform type %s)",
                  "Phase transition coil misalignment"],
        "subsystems": ["Emitter array", "Buffer matrix", "Targeting scanner"],
        "severity": lambda: random.choice(["Level 3 diagnostic required", "Safety protocol override"])
    },
    "structural integrity": {
        "issues": ["Microfractures detected in %s", "Hull polarization failure (Sector %s)",
                  "Structural field stress at %s GPa"],
        "subsystems": ["Primary hull", "Secondary hull", "Nacelle pylons"]
    },
    "computer core": {
        "issues": ["Isolinear burnout in %s", "Positronic network fragmentation",
                  "Subprocessor thermal limit exceeded (%s°C)", "Corrupted memory engram"],
        "subsystems": ["Primary core", "Backup core", "Bridge module"]
    },
    "weapons": {
        "issues": ["Phaser emitter degradation (%s%% output)", "Photon torpedo guidance variance",
                  "Polarized plating capacitor drain"],
        "subsystems": ["Forward array", "Ventral array", "Torpedo bay"]
    },
    "environmental": {
        "issues": ["Atmospheric mix variance (%s ppm)", "Grav plating oscillation (+/- %s%%)",
                  "Thermal regulation failure (%s°C deviation)"],
        "subsystems": ["Life support", "Replicators", "Waste reclamation"]
    },
    "propulsion": {
        "issues": ["Impulse manifold plasma leak", "Thruster fuel imbalance (%s:1 ratio)",
                  "RCS venturi clogged"],
        "subsystems": ["Port thrusters", "Starboard thrusters", "Dorsal thrusters"]
    },
    "sensors": {
        "issues": ["Tachyon detection threshold breach", "Subspace interference pattern %s",
                  "Multispectral analysis grid failure"],
        "subsystems": ["Long-range", "Tactical", "Astrometric"]
    }
}

# ===== GENERATION FUNCTIONS =====
def generate_stardate(era="TNG"):
    base = {"TOS": 2000, "TNG": 41000, "DS9": 47000, "VOY": 49000}[era]
    return f"{base + random.randint(0,9999):05}.{random.randint(1,9)}"

def generate_crew_name():
    rank = random.choice(["Ensign", "Lt.", "Lt. Cmdr.", "Cmdr.", "Captain"])
    return f"{rank} {fake.last_name()} ({random.choice(DEPARTMENTS)})"

def generate_cross_ref():
    if random.random() > 0.6:
        return f"\nCrossref: Similar anomalies logged at {random.choice(STARBASE_NAMES)} (Stardate {generate_stardate()})"
    return ""

def generate_issue(system):
    config = SYSTEMS[system]
    issue = random.choice(config["issues"])
    
    # Dynamic value insertion
    if "%s" in issue:
        if "harmonic" in issue: val = f"{random.uniform(0.7, 1.3):.2f}Δ"
        elif "ratio" in issue: val = f"{random.randint(2,9)}:{random.randint(1,5)}"
        elif "ppm" in issue: val = f"{random.randint(50, 5000)}"
        elif "°C" in issue: val = f"{random.randint(5, 150)}"
        else: val = random.choice(DECK_GRID)
        issue = issue % val
    
    subsystem = f"({random.choice(config['subsystems'])}) " if random.random() > 0.4 else ""
    return f"{subsystem}{issue}"

# ===== REPORT GENERATION =====
def generate_report():
    num_issues = np.random.poisson(lam=3) + 1  # Realistic issue clustering
    systems = random.sample(list(SYSTEMS.keys()), k=min(num_issues, len(SYSTEMS)))
    
    report = {
        "title": f"USS Enterprise-{random.choice(['D','E','NCC-1701'])} Status Report",
        "stardate": generate_stardate(),
        "crew": generate_crew_name(),
        "issues": [generate_issue(sys) for sys in systems],
        "systems": systems,
        "addendum": generate_cross_ref()
    }
    
    return report

# ===== OUTPUT FORMATTING =====
def format_report(report):
    return f"""begin note
----------------
{report['title']}
Notes by {report['crew']}, Stardate {report['stardate']}

Inspection notes:
{', '.join(report['issues']).capitalize()}.{report['addendum']}
------------------
Affected components:
{';'.join(report['systems'])}
"""

# Generate 200 reports
for _ in range(200):
    print(format_report(generate_report()))
    print("\n")