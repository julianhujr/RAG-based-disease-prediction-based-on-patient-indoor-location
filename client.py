import requests
import json
import os
import sys
from typing import Optional

class MedicalAnalysisClient:
    def __init__(self, server_url: str = "http://localhost:8000"):
        """
        Initialize the medical analysis client
        
        Args:
            server_url: URL of the FastAPI server
        """
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
    
    def check_server_health(self) -> bool:
        """Check if the server is running and healthy"""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def analyze_data(self, json_file_path: str) -> Optional[dict]:
        """
        Send JSON data file to server for analysis
        
        Args:
            json_file_path: Path to the JSON data file
            
        Returns:
            Analysis results as dictionary or None if failed
        """
        if not os.path.exists(json_file_path):
            print(f"Error: File {json_file_path} does not exist")
            return None
        
        if not json_file_path.endswith('.json'):
            print(f"Error: File must be a JSON file")
            return None
        
        try:
            # Check server health first
            if not self.check_server_health():
                print("Error: Server is not responding. Make sure the FastAPI server is running.")
                return None
            
            print(f"Uploading {json_file_path} to server for analysis...")
            
            with open(json_file_path, 'rb') as file:
                files = {'file': (os.path.basename(json_file_path), file, 'application/json')}
                
                response = self.session.post(
                    f"{self.server_url}/analyze",
                    files=files,
                    timeout=300  # 5 minutes timeout for analysis
                )
            
            if response.status_code == 200:
                print("Analysis completed successfully!")
                return response.json()
            else:
                print(f"Analysis failed with status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("Error: Request timed out. Analysis may take longer than expected.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error: Request failed - {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return None
    
    def save_results(self, results: dict, output_path: str = "output.json") -> bool:
        """
        Save analysis results to a JSON file
        
        Args:
            results: Analysis results dictionary
            output_path: Path to save the results
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return False
    
    def print_summary(self, results: dict):
        """Print a summary of the analysis results"""
        if not results:
            print("No results to display")
            return
        
        print("\n" + "="*80)
        print("MEDICAL ANALYSIS SUMMARY")
        print("="*80)
        
        metadata = results.get('analysis_metadata', {})
        symptoms = results.get('enhanced_symptoms', [])
        diseases = results.get('disease_predictions', [])
        
        print(f"Analysis Date: {metadata.get('analysis_date', 'Unknown')}")
        
        data_range = metadata.get('data_date_range', {})
        print(f"Data Range: {data_range.get('start', 'Unknown')} to {data_range.get('end', 'Unknown')}")
        print(f"Total Days: {data_range.get('total_days', 0)}")
        
        models = metadata.get('models_used', {})
        print(f"Symptom Detection Model: {models.get('symptom_detection', 'Unknown')}")
        print(f"Disease Prediction Model: {models.get('disease_prediction', 'Unknown')}")
        
        print(f"\nSYMPTOMS DETECTED: {len(symptoms)}")
        print("-" * 50)
        for i, symptom in enumerate(symptoms, 1):
            print(f"{i}. {symptom.get('Abnormal Activity', 'Unknown')}")
            print(f"   Confidence: {symptom.get('confidence', 'Unknown')}")
            definition = symptom.get('definition', 'Not provided')
            if len(definition) > 100:
                definition = definition[:100] + "..."
            print(f"   Definition: {definition}")
            print()
        
        print(f"DISEASE PREDICTIONS: {len(diseases)}")
        print("-" * 50)
        for i, disease in enumerate(diseases, 1):
            print(f"{i}. {disease.get('disease', 'Unknown')}")
            print(f"   Confidence: {disease.get('confidence', 'Unknown')}")
            print(f"   Related Symptoms: {len(disease.get('related_abnormal_activities', []))}")
            
            # Highlight high-priority conditions
            disease_name = disease.get('disease', '').lower()
            if any(term in disease_name for term in ['dementia', 'alzheimer', 'cognitive', 'parkinson', 'cancer']):
                print(f"   *** HIGH PRIORITY CONDITION ***")
            
            related_activities = disease.get('related_abnormal_activities', [])
            if related_activities:
                print(f"   Symptoms Explained: {', '.join(related_activities[:3])}")
                if len(related_activities) > 3:
                    print(f"   ... and {len(related_activities) - 3} more")
            print()
        
        print("="*80)

def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 2:
        print("Usage: python client.py <input_json_file> [output_json_file] [server_url]")
        print("Example: python client.py data/data1.json output.json http://localhost:8000")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output.json"
    server_url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8000"
    
    # Initialize client
    client = MedicalAnalysisClient(server_url)
    
    print("Medical Analysis Client")
    print(f"Server: {server_url}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print("-" * 50)
    
    # Analyze data
    results = client.analyze_data(input_file)
    
    if results:
        # Save results
        if client.save_results(results, output_file):
            # Print summary
            client.print_summary(results)
            print(f"\nAnalysis complete! Results saved to {output_file}")
        else:
            print("Failed to save results")
    else:
        print("Analysis failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()