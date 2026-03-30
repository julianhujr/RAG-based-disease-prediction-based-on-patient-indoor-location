import torch
import requests
import json
import re
import numpy as np
from collections import defaultdict
from auto_embed import AutoEmbedding
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

class LLMJudge:
    def __init__(self, config):
        self.config = config
        self.openrouter_api_key = "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"
        self.openrouter_api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "openai/gpt-5-mini"  # Using same model as the predictor
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def call_llm(self, prompt, temperature=0.2, timeout=30, max_retries=3):
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": "You are a precise medical evaluation assistant with expertise in clinical reasoning and differential diagnosis. Always respond in valid JSON format as specified, with no additional text outside the JSON structure."},
                {"role": "user", "content": prompt}
            ]
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.openrouter_api_url, 
                    headers=headers, 
                    json=data,
                    timeout=timeout
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"].strip()
                # Clean up markdown formatting
                cleaned_content = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.MULTILINE).strip()
                return cleaned_content
            except (requests.RequestException, requests.Timeout) as e:
                if attempt < max_retries - 1:
                    print(f"API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    continue
                else:
                    raise Exception(f"API call failed after {max_retries} attempts: {str(e)}")

    def enhanced_disease_symptom_retrieval(self, disease, retriever):
        """Enhanced retrieval for disease symptoms with multiple query strategies"""
        queries = [
            f"Common symptoms and signs of {disease}",
            f"Clinical presentation of {disease}",
            f"Diagnostic criteria for {disease}",
            f"{disease} symptoms manifestations",
            f"How to diagnose {disease} symptoms"
        ]
        
        all_docs = []
        seen_content = set()
        
        for query in queries:
            try:
                retrieved_docs = retriever.invoke(query)
                for doc in retrieved_docs:
                    content_hash = hash(doc.page_content[:300])
                    if content_hash not in seen_content:
                        all_docs.append(doc)
                        seen_content.add(content_hash)
            except Exception as e:
                print(f"Warning: Retrieval failed for query '{query}': {str(e)}")
                continue
        
        # Rerank documents
        if all_docs:
            pairs = [[f"Symptoms and clinical features of {disease}", doc.page_content] for doc in all_docs]
            scores = self.reranker.predict(pairs)
            sorted_indices = np.argsort(scores)[::-1][:12]  # Top 12 documents
            return [all_docs[i] for i in sorted_indices]
        
        return []

    def extract_disease_symptoms(self, disease, context):
        """Use LLM to extract structured symptoms from medical context"""
        extract_prompt = f"""
Analyze the following medical literature context and extract the most common and characteristic symptoms/signs for {disease}.

Medical Context:
{context}

Instructions:
1. Extract 8-12 most common symptoms/signs mentioned in the context
2. Focus on symptoms that are characteristic or frequently mentioned
3. Use clear, medical terminology
4. Avoid general terms like "discomfort" - be specific
5. Include both subjective symptoms (what patient feels) and objective signs (what can be observed)

Output only in JSON format:
{{"symptoms": ["Specific symptom 1", "Specific symptom 2", ...]}}
"""
        try:
            response = self.call_llm(extract_prompt, temperature=0.1)
            symptoms_data = json.loads(response)
            return symptoms_data.get("symptoms", [])
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Failed to extract symptoms for {disease}: {str(e)}")
            return []

    def calculate_symptom_overlap(self, patient_symptoms, disease_symptoms):
        """Calculate overlap between patient symptoms and disease symptoms"""
        if not patient_symptoms or not disease_symptoms:
            return 0.0, []
        
        # Normalize symptoms for comparison (lowercase, remove extra spaces)
        normalized_patient = [s.lower().strip() for s in patient_symptoms]
        normalized_disease = [s.lower().strip() for s in disease_symptoms]
        
        matches = []
        for p_symptom in normalized_patient:
            for d_symptom in normalized_disease:
                # Check for partial matches (key terms)
                if self.symptoms_match(p_symptom, d_symptom):
                    matches.append((p_symptom, d_symptom))
                    break
        
        overlap_percentage = (len(matches) / len(normalized_patient)) * 100
        return overlap_percentage, matches

    def symptoms_match(self, patient_symptom, disease_symptom):
        """Check if patient symptom matches disease symptom"""
        # Simple keyword matching - can be enhanced
        patient_words = set(patient_symptom.lower().split())
        disease_words = set(disease_symptom.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        patient_words -= stop_words
        disease_words -= stop_words
        
        # Check for significant overlap
        if len(patient_words & disease_words) >= 2:  # At least 2 common words
            return True
        
        # Check for specific medical term matches
        medical_terms = {
            'bleeding': ['blood', 'hemorrhage'],
            'pain': ['ache', 'discomfort', 'cramping'],
            'bowel': ['stool', 'defecation', 'evacuation'],
            'sadness': ['depression', 'depressed', 'mood'],
            'anhedonia': ['pleasure', 'interest', 'enjoyment']
        }
        
        for patient_word in patient_words:
            for disease_word in disease_words:
                if patient_word == disease_word:
                    return True
                # Check synonyms
                for term, synonyms in medical_terms.items():
                    if (patient_word == term and disease_word in synonyms) or \
                       (disease_word == term and patient_word in synonyms):
                        return True
        
        return False

    def judge_predictions(self, symptom_list, predictions):
        print("Loading RAG database for judgment...")
        
        # Load RAG configuration and embeddings
        rag_config = self.config["rag_db"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embeddings = AutoEmbedding(rag_config["model_name"], rag_config["embedding_type"], model_kwargs={"device": device})
        
        try:
            rag_db = FAISS.load_local(rag_config["faiss_path"], embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            raise Exception(f"Failed to load FAISS database: {str(e)}")

        # Set up retriever
        retriever = rag_db.as_retriever(search_kwargs={"k": 25})

        print("Retrieving disease symptom information...")
        
        # Extract disease symptoms from RAG database
        disease_symptoms = {}
        disease_predictions = predictions.get("disease_predictions", predictions.get("Potential Disease Prediction", []))
        
        for i, pred in enumerate(disease_predictions):
            disease = pred.get("disease", pred.get("Disease", ""))
            print(f"Processing disease {i+1}/{len(disease_predictions)}: {disease}")
            
            # Enhanced retrieval for this disease
            top_docs = self.enhanced_disease_symptom_retrieval(disease, retriever)
            
            if not top_docs:
                print(f"Warning: No documents retrieved for disease: {disease}")
                disease_symptoms[disease] = []
                continue
            
            # Build context from retrieved documents
            context = ""
            for j, doc in enumerate(top_docs):
                context += f"\n\n--- Medical Reference {j+1} for {disease} ---\n"
                context += doc.page_content.strip()
            
            # Extract symptoms using LLM
            symptoms = self.extract_disease_symptoms(disease, context)
            disease_symptoms[disease] = symptoms

        print("Performing comprehensive judgment analysis...")
        
        # Prepare data for judgment
        symptoms_str = "\n".join([f"{i+1}. {sym}" for i, sym in enumerate(symptom_list)])
        predictions_str = json.dumps(disease_predictions, indent=2)
        disease_symptoms_str = json.dumps(disease_symptoms, indent=2)
        
        # Enhanced judgment criteria
        criteria = """
EVALUATION CRITERIA:

1. SYMPTOM OVERLAP ANALYSIS:
   - Calculate exact overlap between patient symptoms and retrieved disease symptoms
   - High overlap (≥60%): Supports "Very Likely" or "Likely"
   - Moderate overlap (30-59%): Supports "Possible" or "Likely"
   - Low overlap (10-29%): Supports "Unlikely"
   - Very low overlap (<10%): Supports "Very Unlikely"

2. RELATED ACTIVITIES VALIDATION:
   - Verify all listed symptoms are exactly from patient's symptom list
   - Check for any fabricated or modified symptoms
   - Ensure no important patient symptoms are missed for the disease

3. REASONING QUALITY:
   - Medical accuracy and evidence-based reasoning
   - Proper use of retrieved medical context
   - Logical connection between symptoms and disease
   - Absence of unsupported medical claims

4. CONFIDENCE CALIBRATION:
   - Alignment between confidence level and actual evidence
   - Consider symptom overlap, disease severity, and medical logic
   - Flag overconfident or underconfident predictions

5. CLINICAL REASONABLENESS:
   - Overall medical plausibility
   - Appropriate differential diagnosis approach
   - Consideration of disease prevalence and patient presentation
"""

        judge_prompt = f"""
You are evaluating disease predictions made by a medical AI system. Perform a comprehensive analysis using the specified criteria.

PATIENT SYMPTOMS:
{symptoms_str}

DISEASE PREDICTIONS TO EVALUATE:
{predictions_str}

RETRIEVED DISEASE SYMPTOMS FROM MEDICAL LITERATURE:
{disease_symptoms_str}

EVALUATION CRITERIA:
{criteria}

For each disease prediction, provide:

1. Calculate exact symptom overlap percentage
2. Validate the related_abnormal_activities list
3. Assess reasoning quality and medical accuracy
4. Evaluate confidence level appropriateness
5. Provide overall reasonableness judgment

OUTPUT FORMAT (JSON only):
{{
  "evaluation_results": [
    {{
      "disease": "Disease Name",
      "overlap_percentage": 45.5,
      "overlap_details": ["matched symptom pairs"],
      "related_activities_validation": "Assessment of symptom list accuracy",
      "reasoning_quality": "Assessment of medical reasoning",
      "confidence_assessment": "Analysis of confidence appropriateness with suggested level if different",
      "overall_reasonableness": "Excellent/Good/Fair/Poor",
      "detailed_feedback": "Comprehensive explanation of strengths and weaknesses",
      "recommended_confidence": "Suggested confidence level if different"
    }}
  ],
  "overall_assessment": {{
    "summary_score": 8.2,
    "total_diseases_evaluated": 6,
    "well_justified_predictions": 4,
    "questionable_predictions": 2,
    "major_concerns": ["List any significant issues"],
    "strengths": ["List prediction strengths"],
    "improvement_recommendations": ["Specific recommendations for better predictions"]
  }}
}}
"""

        # Get LLM judgment
        try:
            judge_response = self.call_llm(judge_prompt, temperature=0.1)
            judge_json = json.loads(judge_response)
            
            # Add symptom overlap calculations for verification
            for i, result in enumerate(judge_json.get("evaluation_results", [])):
                disease = result.get("disease", "")
                if disease in disease_symptoms:
                    pred_symptoms = []
                    for pred in disease_predictions:
                        if pred.get("disease", pred.get("Disease", "")) == disease:
                            pred_symptoms = pred.get("related_abnormal_activities", [])
                            break
                    
                    overlap_pct, matches = self.calculate_symptom_overlap(pred_symptoms, disease_symptoms[disease])
                    result["calculated_overlap"] = overlap_pct
                    result["symptom_matches"] = [f"{m[0]} <-> {m[1]}" for m in matches]
            
            return judge_json
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            try:
                # Try to fix common JSON issues
                fixed_response = re.sub(r',\s*}', '}', judge_response)
                fixed_response = re.sub(r',\s*]', ']', fixed_response)
                return json.loads(fixed_response)
            except:
                raise ValueError(f"Invalid JSON response from LLM judge: {judge_response}")

if __name__ == "__main__":
    config = {
        "rag_db": {
            "faiss_path": "./rag",
            "docs_path": "./ocr/output",
            "model_name": "pritamdeka/S-PubMedBert-MS-MARCO",
            "embedding_type": "sentence_transformer",
            "chunk_size": 1024,
            "chunk_overlap": 256
        }
    }
    
    symptom_list = [
        "Severe Sleep Fragmentation and Circadian Rhythm Disruption",
        "Pathological Bathroom Usage Pattern Suggestive of Medical Condition",
        "Disordered Eating Patterns with Possible Binge-Eating Characteristics",
        "Severe Social Isolation and Sedentary Behavior Pattern",
        "Repetitive Compulsive-Like Behavioral Patterns"
    ]
    
    print("Initializing LLM Judge System...")
    
    try:
        # Load the predictions file
        try:
            with open("medical_analysis_results_2025-11-10.json", "r") as f:
                predictions = json.load(f)
        except FileNotFoundError:
            print("Error: improved_output.json not found. Please run the medical RAG system first.")
            exit(1)
        except Exception as e:
            print(f"Error loading predictions file: {str(e)}")
            exit(1)

        judge = LLMJudge(config)
        print("Starting comprehensive evaluation...")
        
        result = judge.judge_predictions(symptom_list, predictions)
        
        # Save results
        with open("judge_evaluation.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE!")
        print("="*60)
        
        # Display summary
        overall = result.get("overall_assessment", {})
        print(f"✓ Summary Score: {overall.get('summary_score', 'N/A')}/10")
        print(f"✓ Total Diseases Evaluated: {overall.get('total_diseases_evaluated', 'N/A')}")
        print(f"✓ Well-Justified Predictions: {overall.get('well_justified_predictions', 'N/A')}")
        print(f"✓ Questionable Predictions: {overall.get('questionable_predictions', 'N/A')}")
        print(f"✓ Results saved to: judge_evaluation.json")
        
        # Show individual results
        print(f"\nDETAILED EVALUATION RESULTS:")
        print("-" * 50)
        for i, result in enumerate(result.get("evaluation_results", []), 1):
            print(f"{i}. {result.get('disease', 'Unknown Disease')}")
            print(f"   Overlap: {result.get('overlap_percentage', 'N/A')}%")
            print(f"   Reasonableness: {result.get('overall_reasonableness', 'N/A')}")
            if result.get('recommended_confidence'):
                print(f"   Recommended Confidence: {result['recommended_confidence']}")
            print()
        
        # Show recommendations
        if overall.get('improvement_recommendations'):
            print("IMPROVEMENT RECOMMENDATIONS:")
            print("-" * 30)
            for rec in overall['improvement_recommendations']:
                print(f"• {rec}")
            
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nTroubleshooting suggestions:")
        print("1. Ensure improved_output.json exists (run the medical RAG system first)")
        print("2. Check your internet connection")
        print("3. Verify the OpenRouter API key is valid")
        print("4. Ensure FAISS database exists at the specified path")