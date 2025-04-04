import random
from faker import Faker
from datetime import datetime

fake = Faker()

# Component configurations
COMPONENTS = {
    "warp drive": ["plasma conduit fluctuations", "dilithium chamber resonance decay",
                  "Bussard collector debris", "warp core coolant leak"],
    "transporter": ["Heisenberg compensator drift", "pattern buffer fragmentation",
                   "molecular imaging scanner glitches"],
    "main computer": ["duotronic subprocessor overheating", "corrupted memory matrices",
                     "isolinear chip latency"],
    "SIF": ["hull microfractures", "structural field harmonics misalignment"],
    "life support": ["CO₂ scrubber lag", "atmospheric pressure variance",
                    "gravity plating oscillations"],
    "deflector": ["shield polarization drift", "navigational deflector burn marks"],
    "phasers": ["power coupling erosion", "phaser capacitor depletion"],
    "impulse engines": ["fusion reactor exhaust variance", "thruster manifold instability"]
}

def generate_stardate():
    return f"{random.randint(54000, 63000):05}.{random.randint(1,9)}"

def generate_report():
    # Select 3-5 random components
    selected = random.sample(list(COMPONENTS.keys()), k=random.randint(2,6))
    issues = [random.choice(COMPONENTS[comp]) for comp in selected]

    # Build note text
    note_text = ", ".join(issues) + "."
    if random.random() > 0.7:  # 30% chance for extra remark
        note_text += f" Additional anomalies logged in sector {random.choice(['Alpha','Beta','Gamma'])}-{random.randint(1,12)}. "

    return {
        "name": f"{fake.name()}, {random.choice(['Starfleet Engineering', 'Starfleet Science Corps', 'Temporal Integrity Commission'])}",
        "stardate": generate_stardate(),
        "notes": note_text,
        "components": ";".join(selected)
    }

if __name__ == '__main__':

    # Generate sample output
    for _ in range(50):  # Change number for desired count
        report = generate_report()
        print(f"""begin note
    ----------------
    USS Enterprise status report
    Notes by {report['name']}, Stardate {report['stardate']}
    Inspection notes:
    {report['notes']}
    ------------------
    Affected components:
    {report['components']}
    """)
