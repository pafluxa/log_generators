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
from typing import Dict, List, Tuple, TypedDict
from dataclasses import dataclass
import numpy as np
from faker import Faker
import textwrap


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
class EnterpriseDiagnosticGenerator:
    """
    Generates USS Enterprise diagnostic reports.

    Features:
    - 36 fully-defined systems with complete cross-references
    - Cascading failure simulations
    - Training-ready output format
    """

    def __init__(self):
        """Initialize with Starfleet-standard configurations."""
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

    def _initialize_system_templates(self) -> Dict[str, SystemTemplate]:
        """Initialize all 36 systems with complete configuration and validation."""
        systems = {
            # Core Systems (12)
            "warp_core": {
                "phrases": ["unstable %s in %s", "%s fluctuations detected"],
                "subsystems": ["plasma conduit", "dilithium matrix", "antimatter flow"],
                "units": ["cochranes", "terawatts", "milliisotons"],
                "fails": ["impulse_drive", "sensors", "artificial_gravity"]
            },
            "transporter_buffer": {
                "phrases": ["%s pattern degradation", "quantum %s detected"],
                "subsystems": ["Heisenberg compensator", "pattern enhancer", "phase discriminator"],
                "units": ["quantum%", "microrems"],
                "fails": ["replicators", "holodeck_safety"]
            },
            "deflector_dish": {
                "phrases": ["%s polarization loss", "%s frequency drift"],
                "subsystems": ["graviton emitter", "metaphasic shielding", "particle deflector"],
                "units": ["gigajoules", "nanometers"],
                "fails": ["navigation", "shield_grid"]
            },
            "impulse_drive": {
                "phrases": ["%s manifold %s", "plasma %s in %s chamber"],
                "subsystems": ["fusion reactors", "driver coils", "magnetic constrictors"],
                "units": ["terawatts", "cochranes", "newtons"],
                "fails": ["thrusters", "power_distribution"]
            },
            "shield_grid": {
                "phrases": ["%s modulation failure", "%s frequency %s"],
                "subsystems": ["graviton generators", "phase variance emitters", "multiphasic nodes"],
                "units": ["gigahertz", "isotons", "percent"],
                "fails": ["secondary_hull_plating", "deflector_dish"]
            },
            "structural_integrity": {
                "phrases": ["%s stress fractures", "%s field %s"],
                "subsystems": ["duranium grid", "polarized plating", "ablative generators"],
                "units": ["GPa", "newtons/sq.m"],
                "fails": ["inertial_dampers", "atmospheric_containment"]
            },
            "atmospheric_containment": {
                "phrases": ["%s dropping", "%s pressure %s"],
                "subsystems": ["pressure membranes", "primary_hull", "secondary_hull_plating"],
                "units": ["GPa", "newtons/sq.m"],
                "fails": ["secondary_hull_plating"]
            },
            "computer_core": {
                "phrases": ["%s processing %s", "memory %s fragmentation"],
                "subsystems": ["isolinear chips", "optical data network", "quantum processors"],
                "units": ["gigaquads", "nanoseconds", "petaflops"],
                "fails": ["ops_systems", "security_protocols"]
            },
            "sensors": {
                "phrases": ["%s calibration %s", "%s interference %s"],
                "subsystems": ["tachyon detectors", "gravimetric scanners", "subspace beacons"],
                "units": ["arcseconds", "parsecs", "light-years"],
                "fails": ["navigation", "tactical_systems"]
            },
            "coolant_system": {
                "phrases": ["%s viscosity anomaly", "thermal %s failure"],
                "subsystems": ["deuterium pumps", "heat exchangers", "emergency vents"],
                "units": ["kelvins", "liters/min"],
                "fails": ["warp_core", "environmental"]
            },
            "environmental": {
                "phrases": ["%s controls %s", "atmospheric %s collapse"],
                "subsystems": ["oxygen mixers", "thermal regulators", "pressure membranes"],
                "units": ["ppm", "pascals", "CFM"],
                "fails": ["life_support", "crew_quarters"]
            },
            "navigation": {
                "phrases": ["%s alignment %s", "stellar %s drift"],
                "subsystems": ["astrometric processors", "warp field calculators", "impulse vectoring"],
                "units": ["arcminutes", "parsecs", "cubic Warp"],
                "fails": ["helm_control", "sensor_palette"]
            },
            "power_distribution": {
                "phrases": ["%s relay %s", "EPS %s overload"],
                "subsystems": ["plasma conduits", "electro-inductors", "capacitor banks"],
                "units": ["terawatts", "volts", "ohms"],
                "fails": ["warp_core", "auxiliary_power"]
            },

            # Medical Systems (4)
            "medical": {
                "phrases": ["%s scanner %s", "biofilter %s overload"],
                "subsystems": ["tricorder arrays", "hypospray synthesizers", "surgical holograms"],
                "units": ["ccs", "millisieverts", "microbes"],
                "fails": ["sickbay", "quarantine_fields"]
            },
            "quarantine_fields": {
                "phrases": ["%s containment %s", "force %s degradation"],
                "subsystems": ["level-10 forcefields", "bio-dampers", "decon emitters"],
                "units": ["microbes", "quads", "sterilization%"],
                "fails": ["security_protocols", "environmental"]
            },
            "decon_protocols": {
                "phrases": ["%s sequence %s", "sterilization %s failure"],
                "subsystems": ["antimicrobial fields", "radiation sweepers", "nanoprobe scrubbers"],
                "units": ["rads", "microbes", "decon%"],
                "fails": ["medical", "crew_quarters"]
            },
            "bio-neural_gel_packs": {
                "phrases": ["%s synaptic %s", "neural %s degradation"],
                "subsystems": ["isolinear nodes", "neuro-electric pathways", "protein matrices"],
                "units": ["quads", "synapses/sec", "nanobots"],
                "fails": ["computer_core", "medical"]
            },

            # Tactical Systems (6)
            "phaser_banks": {
                "phrases": ["%s capacitor %s", "%s emitter %s"],
                "subsystems": ["nadion generators", "phase coils", "energy relays"],
                "units": ["megajoules", "volts", "amperes"],
                "fails": ["targeting_sensors", "power_core"]
            },
            "torpedo_launchers": {
                "phrases": ["%s loading %s", "%s guidance %s"],
                "subsystems": ["antimatter pods", "magnetic rails", "targeting scanners"],
                "units": ["isotons", "gauss", "radians"],
                "fails": ["weapons_locker", "defensive_systems"]
            },
            "tactical_systems": {
                "phrases": ["%s algorithm %s", "threat %s overload"],
                "subsystems": ["predictive matrices", "weapon interlocks", "shield modulation"],
                "units": ["T-flops", "nanoseconds", "targets"],
                "fails": ["phaser_banks", "defensive_systems"]
            },
            "defensive_systems": {
                "phrases": ["%s protocol %s", "countermeasure %s failure"],
                "subsystems": ["electronic warfare", "point-defense", "cloak detection"],
                "units": ["gigahertz", "decibels", "interception%"],
                "fails": ["shield_grid", "tactical_systems"]
            },
            "weapons_locker": {
                "phrases": ["%s containment %s", "security %s breach"],
                "subsystems": ["phaser banks", "photon torpedoes", "security forcefields"],
                "units": ["units", "security_level", "authorization%"],
                "fails": ["security_protocols", "armory"]
            },
            "security_protocols": {
                "phrases": ["%s override %s", "access %s violation"],
                "subsystems": ["biometric scanners", "forcefield emitters", "intrusion detection"],
                "units": ["security_level", "unauthorized_accesses", "breaches"],
                "fails": ["computer_core", "bridge_security"]
            },

            # Engineering Systems (6)
            "bussard_collectors": {
                "phrases": ["%s hydrogen %s", "interstellar %s contamination"],
                "subsystems": ["ram-scoop fields", "particle filters", "magnetic conduits"],
                "units": ["grams/sec", "isotons", "gauss"],
                "fails": ["warp_core", "fuel_processing"]
            },
            "tractor_beam": {
                "phrases": ["%s emitter %s", "gravimetric %s distortion"],
                "subsystems": ["focusing coils", "spatial projectors", "attractor arrays"],
                "units": ["newtons", "gravons", "picochranes"],
                "fails": ["deflector_dish", "inertial_dampers"]
            },
            "replicators": {
                "phrases": ["%s pattern %s", "matter %s corruption"],
                "subsystems": ["quantum templates", "molecular assemblers", "waste reclamators"],
                "units": ["grams/sec", "quads", "atom%"],
                "fails": ["transporter_buffer", "crew_mess"]
            },
            "holodeck_safety": {
                "phrases": ["%s protocol %s", "simulation %s failure"],
                "subsystems": ["photonic buffers", "matter replicators", "morphogenic projectors"],
                "units": ["quads", "kilobytes", "nanobots"],
                "fails": ["life_support", "computer_core"]
            },
            "inertial_dampers": {
                "phrases": ["%s matrix %s", "%s compensation %s"],
                "subsystems": ["graviton emitters", "acceleration buffers", "gyroscopic stabilizers"],
                "units": ["Gs", "newtons", "pascals"],
                "fails": ["artificial_gravity", "structural_integrity"]
            },
            "artificial_gravity": {
                "phrases": ["%s grid %s", "gravimetric %s fluctuations"],
                "subsystems": ["gravity plates", "inertial compensators", "mass sensors"],
                "units": ["Gs", "newtons", "pascals"],
                "fails": ["inertial_dampers", "structural_integrity"]
            },
            # Auxiliary Systems (8)
            "thrusters": {
                "phrases": ["%s vector %s", "maneuvering %s failure"],
                "subsystems": ["ion pods", "reaction control", "vernier adjustments"],
                "units": ["newtons", "degrees/sec", "impulse%"],
                "fails": ["navigation", "structural_integrity"]
            },
            "long_range_comms": {
                "phrases": ["%s subspace %s", "carrier %s distortion"],
                "subsystems": ["hyperwave transceivers", "quasar amplifiers", "antique radio"],
                "units": ["warp-factors", "parsecs", "gigaquads"],
                "fails": ["bridge_module", "emergency_beacons"]
            },
            "warp_field_stabilizers": {
                "phrases": ["%s geometry %s", "subspace %s collapse"],
                "subsystems": ["graviton emitters", "quantum flux regulators", "manifold projectors"],
                "units": ["cochranes", "millicochranes", "warp-factors"],
                "fails": ["navigational_sensors", "impulse_drive"]
            },
            "emergency_medical_hologram": {
                "phrases": ["%s matrix %s", "holo-%s degradation"],
                "subsystems": ["photonic processors", "medical databases", "ethical subroutines"],
                "units": ["gigaquads", "nanobots", "bedside%"],
                "fails": ["sickbay", "computer_core"]
            },
            "sensor_palette": {
                "phrases": ["%s calibration %s", "multi-%s interference"],
                "subsystems": ["tachyon beams", "gravimetric arrays", "bio-scanners"],
                "units": ["arcminutes", "terawatts", "picoamps"],
                "fails": ["science_labs", "tactical_systems"]
            },
            "secondary_hull_plating": {
                "phrases": ["%s microfractures %s", "ablative %s failure"],
                "subsystems": ["tritanium layers", "polarized mesh", "metaphasic coating"],
                "units": ["centimeters", "newtons", "GPa"],
                "fails": ["primary_hull", "shield_modulation"]
            },
            "auxiliary_power": {
                "phrases": ["%s conduit %s", "emergency %s drain"],
                "subsystems": ["fusion generators", "capacitor banks", "EPS taps"],
                "units": ["megawatts", "joules", "volts"],
                "fails": ["life_support", "emergency_lighting"]
            },
            "quantum_torpedo_bay": {
                "phrases": ["%s loading %s", "warhead %s instability"],
                "subsystems": ["zero-point chambers", "spatial warheads", "launch rails"],
                "units": ["isotons", "quantum%", "gigaquads"],
                "fails": ["weapons_locker", "defensive_systems"]
            },
            "life_support": {
                "phrases": ["%s failure in %s", "%s malfunction detected"],
                "subsystems": ["atmosphere processors", "CO2 scrubbers", "nitrogen regulators"],
                "units": ["ppm", "CFM", "BTUs"],
                "fails": ["crew_quarters", "medical"]
            },
            "crew_quarters": {
                "phrases": ["%s environment %s", "habitat %s failure"],
                "subsystems": ["life support nodes", "thermal regulation", "artificial gravity"],
                "units": ["comfort%", "degrees", "humidity%"],
                "fails": ["environmental", "medical"]
            },
            "ops_systems": {
                "phrases": ["%s protocol %s", "operational %s failure"],
                "subsystems": ["command interfaces", "priority channels", "alert systems"],
                "units": ["priority_level", "alerts", "systems_online%"],
                "fails": ["computer_core", "bridge_module"]
            },
            "science_labs": {
                "phrases": ["%s experiment %s", "research %s containment"],
                "subsystems": ["containment fields", "sensor arrays", "analysis computers"],
                "units": ["experiments", "data_quads", "containment%"],
                "fails": ["sensors", "computer_core"]
            },
            "emergency_lighting": {
                "phrases": ["%s circuit %s", "backup %s failure"],
                "subsystems": ["glow panels", "power relays", "battery backups"],
                "units": ["lumens", "volts", "hours"],
                "fails": ["auxiliary_power", "environmental"]
            },
            "emergency_forcefields": {
                "phrases": ["%s containment %s", "emergency %s failure"],
                "subsystems": ["field emitters", "power nodes", "containment projectors"],
                "units": ["field_strength", "containment%", "watts"],
                "fails": ["structural_integrity", "power_distribution"]
            },
            "sickbay": {
                "phrases": ["%s systems %s", "medical %s offline"],
                "subsystems": ["biobeds", "surgical bay", "diagnostic scanners"],
                "units": ["patients", "ccs", "procedures"],
                "fails": ["medical", "emergency_medical_hologram"]
            },
            "helm_control": {
                "phrases": ["%s response %s", "flight %s failure"],
                "subsystems": ["control interfaces", "navigational links", "thruster controls"],
                "units": ["response_time", "latency", "accuracy%"],
                "fails": ["navigation", "computer_core"]
            },
            "primary_hull": {
                "phrases": ["%s integrity %s", "structural %s compromised"],
                "subsystems": ["tritanium alloy", "support beams", "emergency seals"],
                "units": ["GPa", "stress_factor", "integrity%"],
                "fails": ["structural_integrity", "atmospheric_containment"]
            },
            "bridge_module": {
                "phrases": ["%s command %s", "primary %s failure"],
                "subsystems": ["command chairs", "ops station", "tactical displays"],
                "units": ["priority_level", "systems_online%", "alerts"],
                "fails": ["computer_core", "security_protocols"]
            },
            "emergency_beacons": {
                "phrases": ["%s signal %s", "distress %s failure"],
                "subsystems": ["subspace transmitters", "power cells", "antenna array"],
                "units": ["lightyears", "watts", "signal_strength"],
                "fails": ["long_range_comms", "auxiliary_power"]
            },
            "fuel_processing": {
                "phrases": ["%s refinement %s", "deuterium %s failure"],
                "subsystems": ["slush processors", "purification grids", "storage tanks"],
                "units": ["liters/min", "purity%", "slush_density"],
                "fails": ["warp_core", "impulse_drive"]
            },
            "armory": {
                "phrases": ["%s security %s", "weapons %s breach"],
                "subsystems": ["phaser racks", "security forcefields", "access controls"],
                "units": ["weapons", "security_level", "breaches"],
                "fails": ["security_protocols", "weapons_locker"]
            },
            "bridge_security": {
                "phrases": ["%s protocols %s", "bridge %s compromised"],
                "subsystems": ["forcefields", "security teams", "intrusion detection"],
                "units": ["security_level", "breaches", "response_time"],
                "fails": ["security_protocols", "computer_core"]
            },
            "crew_mess": {
                "phrases": ["%s systems %s", "food %s failure"],
                "subsystems": ["replicator terminals", "dining areas", "waste reclamation"],
                "units": ["meals", "nutrition%", "waste%"],
                "fails": ["replicators", "environmental"]
            },
            "shield_modulation": {
                "phrases": ["%s frequency %s", "deflector %s inversion"],
                "subsystems": ["multiphasic emitters", "graviton projectors", "harmonic dampers"],
                "units": ["gigahertz", "isotons", "phase%"],
                "fails": ["deflector_dish", "tactical_systems"]
            },
            "targeting_sensors": {
                "phrases": ["%s lock %s", "weapons %s failure"],
                "subsystems": ["phaser targeting", "torpedo guidance", "threat analysis"],
                "units": ["accuracy%", "response_time", "targets"],
                "fails": ["tactical_systems", "sensors"]
            },
            "power_core": {
                "phrases": ["%s output %s", "main %s fluctuation"],
                "subsystems": ["plasma conduits", "energy regulators", "EPS taps"],
                "units": ["terawatts", "volts", "stability%"],
                "fails": ["warp_core", "auxiliary_power"]
            },
            "navigational_sensors": {
                "phrases": ["%s alignment %s", "stellar %s drift"],
                "subsystems": ["astrometric scanners", "warp field sensors", "impulse tracking"],
                "units": ["arcminutes", "parsecs", "warp_factor"],
                "fails": ["navigation", "sensors"]
            }
        }

        # Final validation
        for sys_name, config in systems.items():
            for failed_system in config["fails"]:
                if failed_system not in systems:
                    raise ValueError(f"System {failed_system} referenced in {sys_name} but not defined")

        return systems

    def _initialize_failure_chains(self) -> List[Tuple[str, List[str]]]:
        """Define failure sequences with validation."""
        chains = [
            ("plasma_leak", ["warp_core", "coolant_system", "environmental"]),
            ("subspace_anomaly", ["sensors", "navigation", "shield_modulation"]),
            ("biohazard", ["medical", "quarantine_fields", "decon_protocols"]),
            ("power_loss", ["warp_core", "impulse_drive", "auxiliary_power"]),
            ("hull_breach", ["structural_integrity", "atmospheric_containment", "emergency_forcefields"]),
            ("weapons_malfunction", ["phaser_banks", "torpedo_launchers", "tactical_systems"]),
            ("computer_virus", ["computer_core", "ops_systems", "security_protocols"]),
            ("sensor_ghosting", ["sensors", "science_labs", "tactical_systems"])
        ]

        # Validate all systems in chains exist
        for root, systems in chains:
            for sys in systems:
                if sys not in self.system_templates:
                    raise ValueError(f"System {sys} in failure chain not defined")

        return chains

    def _get_location(self) -> str:
        """Generate a random ship location."""
        return random.choice(self.deck_locations) + f", Section {random.randint(1, 42)}"

    def generate_technobabble(self, system: str) -> str:
        """Generate symptom description without system names."""
        try:
            template = self.system_templates[system]
            phrase = random.choice(template["phrases"])

            symptoms = []
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
        symptoms = [f"Initial {root_cause.replace('_', ' ')} detected in {self._get_location()}"]
        affected = []

        for system in systems:
            symptom = self.generate_technobabble(system)
            symptoms.append(f"with secondary {symptom}")

            template = self.system_templates.get(system, {})
            subsystem = next(
                (sub for sub in template.get("subsystems", []) if sub in symptom),
                random.choice(template["subsystems"]) if "subsystems" in template else "unknown_subsystem"
            )
            affected.append((system, subsystem))

        return symptoms, affected

    def generate_report(self) -> StarfleetReport:
        """Generate complete diagnostic report."""
        all_symptoms = []
        all_affected = []

        for _ in range(random.randint(2, 4)):
            symptoms, affected = self.generate_failure_chain()
            all_symptoms.extend(symptoms)
            all_affected.extend(affected)

        # Remove duplicates while preserving order
        seen = set()
        unique_affected = []
        for sys, sub in all_affected:
            if (sys, sub) not in seen:
                seen.add((sys, sub))
                unique_affected.append((sys, sub))

        return {
            "note": " ".join(all_symptoms) + random.choice([".", "."]),
            "systems": ";".join(f"{sys}::{sub}" for sys, sub in unique_affected)
        }

    def generate_full_report(self) -> str:
        """Generate formatted Starfleet report."""
        report = self.generate_report()

        return f"""{'-' * 80}\n{str('*' * 22) + '    USS ENTERPRISE-E DIAGNOSTICS    ' + str('*' * 22)}\n{'-' * 80}\n
Stardate {random.randint(50000,60000)}.{random.randint(1,9)} | Auth. Code Sigma-{random.randint(1,9)}{chr(65+random.randint(0,25))}

Inspection notes:
{textwrap.fill(report['note'], 80)}


Affected components:
{textwrap.fill(report['systems'].replace(' ', '_'), 80)}
"""


if __name__ == "__main__":
    try:
        generator = EnterpriseDiagnosticGenerator()
        print(generator.generate_full_report())
    except Exception as e:
        print(f"Critical failure in report generation: {e}")
        print("Engage emergency engineering protocols!")
