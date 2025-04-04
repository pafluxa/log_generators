import random
import numpy as np
from faker import Faker

fake = Faker()
random.seed(42)
np.random.seed(42)

# ===== GALACTIC ENGINEERING TAXONOMY =====
SYSTEM_ARCHITECTURE = {
    # Format: Main System: [Subsystems]
    "warp_propulsion": ["dilithium_matrix", "plasma_conduits", "Bussard_collectors"],
    "tactical": ["phaser_arrays", "photon_torpedos", "deflector_shields"],
    "klingon_systems": ["bat'leth_circuitry", "disruptor_power", "cloaking_coils"],
    "vulcan_systems": ["katra_resonator", "kolinahr_circuit", "surakian_logic_grid"],
    "borg_systems": ["nanoprobe_tubes", "adaptive_shields", "tactical_analysis_grid"],
    "ferengi_systems": ["latinum_storage", "profit-motive_AI", "self-sealing_stembolts"],
    "cardassian_systems": ["spiral_power", "obsidian_encryption", "terok_nor_sensors"],
    "operations": ["life_support", "artificial_gravity", "replicators"]
}

FAILURE_CASCADES = [
    ("plasma_storm", ["warp_propulsion::dilithium_matrix", "klingon_systems::cloaking_coils", "operations::life_support"]),
    ("borg_assimilation", ["borg_systems::nanoprobe_tubes", "tactical::deflector_shields", "operations::replicators"]),
    ("logic_cascade", ["vulcan_systems::katra_resonator", "cardassian_systems::spiral_power", "operations::artificial_gravity"]),
    ("ferengi_sabotage", ["ferengi_systems::profit-motive_AI", "cardassian_systems::obsidian_encryption", "tactical::photon_torpedos"])
]

SPECIES_PHRASES = {
    "klingon": ["Honor circuits failing at %s", "Disruptor overload in %s quadrant", "Bat'leth resonance %s%% depleted"],
    "vulcan": ["Katra alignment %s illogical", "Surakian matrix %s divergence", "IDIC protocol %s violated"],
    "borg": ["Nanoprobe efficiency %s impaired", "Adaptation cycle %s incomplete", "Tactical analysis %s miscalibrated"],
    "ferengi": ["Profit margin dropped %s bars", "Latinum purity %s subprime", "Rule of Acquisition %s breached"],
    "cardassian": ["Spiral power flux %s critical", "Obsidian protocol %s compromised", "Terok Nor sensors %s offline"]
}

# ===== MULTISPECIES FAILURE GENERATOR =====
def generate_affected_components(chain):
    return [item.split("::")[0] if "::" in item else item for item in chain]

def generate_species_issue(system):
    main, subsystem = (system.split("::") + [None])[:2]
    species = main.split("_")[0] if "_systems" in main else None

    if species in SPECIES_PHRASES:
        phrase = random.choice(SPECIES_PHRASES[species])
        value = f"{random.randint(1,99)}{'%' if 'depleted' in phrase else ''}"
        return phrase.replace("%s", value)
    else:
        tech = ["quantum flux %s", "subspace variance %s", "chroniton surge %s"][random.randint(0,2)]
        return tech.replace("%s", f"0x{random.randint(0x100, 0xFFF):03X}")

def generate_log_entry():
    cascade = random.choice(FAILURE_CASCADES)
    components = [sys.split("::") for sys in cascade[1]]
    systems_in_order = [sys.split("::")[0] for sys in cascade[1]]

    report_lines = [f"Log triggered by a {cascade[0].replace('_', ' ').title()} event. "]

    for sys in cascade[1]:
        main, subsystem = sys.split("::") if "::" in sys else (sys, None)
        report_lines.append(f"{generate_species_issue(sys)}. ")

    return {
        "note": "".join(report_lines),
        "components": ";".join([f"{main}::{sub}" if sub else main for main, sub in components])
    }

# ===== STARFLEET-COMPLIANT OUTPUT =====
def generate_full_report():
    log = generate_log_entry()
    return f"""
USS Enterprise-G Multispecies Diagnostic
Stardate {random.randint(75000,76000)}.{random.randint(1,9)} | Priority {random.choice(['Omega','Sigma','Theta'])}

Log:
{log['note']}

Affected components:
{log['components']}
"""

# Generate 3 sample reports
for _ in range(1000):
    print(generate_full_report())
    print("\n")

# Sample Output:
"""
USS Enterprise-G Multispecies Diagnostic
Stardate 75342.7 | Priority Theta

Inspection notes:
Log triggered by a Borg Assimilation event. Nanoprobe efficiency 78% impaired. Deflector harmonics 0xC4F compromised. Replicator pattern delta-5 corruption

------------------
Affected components:
borg_systems::nanoprobe_tubes;tactical::deflector_shields;operations::replicators
"""
