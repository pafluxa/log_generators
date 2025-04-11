"""
Generates realistic car diagnoses.
"""
import re
import json
import requests

import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    style="%",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.DEBUG,
)
from typing import Dict, List, Tuple, TypedDict, Optional
from dataclasses import dataclass, field


@dataclass(init=True, repr=True, eq=True)
class CarDiagnosticGenerator:
    """
    Generates car diagnostic reports.
    """
    # Predefined list of valid systems (in order of matching priority)
    valid_systems: list = field(default_factory=lambda:
    [
        "Fuel System", "Onboard Computer", "Battery/Charging",  # Multi-word first
        "Engine", "Transmission", "Electrical", "Brakes",
        "Suspension", "Exhaust", "HVAC", "Steering", "Tires"
    ])
    prompt: str = \
"""
Role: Act as an automotive technician drafting a service note for a vehicle.
Task: Describe observed symptoms and customer complaints without naming the affected systems in the main note. At the end, list the systems these symptoms likely relate to.

Instructions for Symptoms Section:

Use non-technical, observable terms (e.g., "grinding noise," "flickering lights").
Avoid jargon (e.g., say "shaking at idle," not "misfire").
Include customer quotes (e.g., "Pedal feels spongy").
Add technician observations (e.g., "Fluid residue near rear axle").
Systems to Consider (Include 5–8 in final list):

Engine, Transmission, Electrical, Brakes, Suspension, Fuel System, Exhaust, HVAC, Steering, Onboard Computer, Battery/Charging, Tires.
Format Requirements:

Header: "Service Note – Reported Symptoms" (bold).
Symptoms/Observations: Bullet points.
Affected Systems List: At the end, under "Likely Affected Systems:" (bold).
No markdown; use plain text with simple symbols (e.g., -, **).
"""
    request_header: Dict[str, str] = field(default_factory=lambda: {"Content-Type": "application/json"})
    model_name: str = ""
    url: str = "localhost"
    port: int = 11424
    server_endpoint: str = ""

    def __post_init__(self):
        """ Initialize logger.
        """
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler("generator.log", mode="a", encoding="utf-8")
        file_handler.setLevel("DEBUG")
        console_handler.setLevel("INFO")
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.formatter = logging.Formatter(
            "{asctime} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M",
        )


    def generate_single_report(self) -> str:
        """ Sends prompt to Ollama server.

            See `template` for checking the structure of the output.

            Returns
            report: dictionary with keys
            - "report": full answer from the server.
        """
        report = ""

        server_endpoint = f"{self.url}:{self.port}/api/generate"

        # create payload with prompt
        self.logger.debug("Building request.")
        content = self.prompt
        payload = {
            "model": self.model_name,
            "prompt": content,
            "stream": False
        }
        self.logger.debug("Initiating comms.")
        max_retries = 10
        retry = 0
        while retry < max_retries:
            self.logger.debug("Waiting for server to answer")
            # send pyload
            response = requests.post(
                server_endpoint,
                headers=self.request_header,
                data=json.dumps(payload)
            )
            self.logger.debug("Response received!")
            # check for server response
            if response.status_code == 200:
                self.logger.debug("Request was answered by server.")
                response_text = response.text
                payload = json.loads(response_text)
                report = payload['response']
                self.logger.debug(f"Server answer: \n\nanswer=%s\n\n", note)
                break
            else:
                self.logger.error(f"Server responded %s with message %s. Retrying.",
                                  response.status_code,
                                  response.text)
                retry = retry + 1
            if retry > max_retries:
                self.logger.error("Exceeded max number of retries. Aborting.")
                raise RuntimeError("Server is not responding.")

        return report

    def parse_report(self, report: str) -> Dict[str, str]:
        
        result = {'report': '', 'systems': ''}
        
        # 1. Extract Symptoms Section (unchanged)
        symptoms_match = re.search(
            r'Service Note – Reported Symptoms\s*([\s\S]*?)(?:\*\*Likely Affected Systems:\*\*|\Z)',
            report,
            re.IGNORECASE
        )
        if symptoms_match:
            symptoms_text = symptoms_match.group(1).strip()
            symptoms_lines = [line.strip('- ').strip() for line in symptoms_text.split('\n') if line.strip()]
            result['report'] = ' '.join(symptoms_lines)
        
        # 2. Brute-force Systems Extraction
        systems_match = re.search(
            r'\*\*Likely Affected Systems:\*\*', report,
            re.IGNORECASE
        )
        
        if systems_match:
            print("MATCH!!")
            systems_text = systems_match.group(1)
            # Convert to lowercase for case-insensitive matches
            clean_text = systems_text.lower()
            # Remove all special characters except spaces and slashes (for Battery/Charging)
            #clean_text = re.sub(r'[^a-z0-9/]+', ' ', systems_text)
            
            found_systems = []
            for system in self.valid_systems:
                # Create search pattern (handle Battery/Charging specially)
                if system == 'Battery/Charging':
                    search_terms = ['battery', 'charging']
                else:
                    search_terms = [system.lower()]
                
                # Check if any search term appears as a whole word
                for term in search_terms:
                    print(term)
                    print(clean_text)
                    if term in clean_text:
                        found_systems.append(system)
                        # break
            result['systems'] = ';'.join(sorted(found_systems))
        
        return result

if __name__ == "__main__":

    import time
    import glob
    import hashlib
    from tqdm import tqdm
    from pathlib import Path

    gen = CarDiagnosticGenerator()

    model_name = 'phi4:14b-fp16'
    # hash = hashlib.sha1(str(time.asctime()).encode("UTF-8")).hexdigest()
    # run_id = hash[0:8]
    run_id = "8262de70"
    gen.logger.info(f"Parsing reports for run {run_id}")
    files = glob.glob(f"./txt/car_reports/{model_name}/{run_id}/server_answers/*.txt")
    for f in tqdm(files):
        report = {}
        with open(f, 'r') as report_file:
            report = gen.parse_report(report_file.read())
            print(f"\nREPORT {report_file}\n")
            print(report['report'])
            print("\nSYSTEMS \n")
            print(report['systems'])
#       Path(f"./txt/{model_name}/{run_id}/reports").mkdir(parents=True, exist_ok=True)
#       # write full report
#       report = generator.generate_report()
#       Path(f"./txt/car_reports/{model_name}/{run_id}/reports/").mkdir(parents=True, exist_ok=True)
#       with open(f"./txt/car_reports/{model_name}/{run_id}/reports/{k:06d}.txt", 'w') as f:
#           f.write(report['report'])
#       
#       k = k + 1
