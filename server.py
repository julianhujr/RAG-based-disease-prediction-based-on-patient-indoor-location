from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import logging
import requests
import torch
import numpy as np
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from auto_embed import AutoEmbedding
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from raw2abtemplate import enhanced_prompt_template
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical Analysis API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IntegratedMedicalAnalysisPipeline:
    def __init__(self, config):
        """
        Integrated pipeline combining data preprocessing, symptom detection, and disease prediction
        """
        # API Configuration for symptom detection
        self.symptom_config = config.get('symptom_detection', {})
        self.openrouter_api_key = self.symptom_config.get('api_key', '......')
        self.openrouter_api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.symptom_model = self.symptom_config.get('model', 'deepseek/deepseek-chat-v3.1')
        self.temperature = self.symptom_config.get('temperature', 0.1)
        self.max_tokens = self.symptom_config.get('max_tokens', 10000)
        
        # Disease prediction configuration
        self.disease_config = config.get('disease_prediction', {})
        self.disease_model = self.disease_config.get('model', "google/gemini-2.5-pro")
        self.rag_config = config.get("rag_db", {})
        
        # Set up headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost:3000",
            "X-Title": "Medical Analysis Pipeline"
        }
        
        # Analysis parameters
        self.analysis_config = {
            'min_confidence_threshold': self.symptom_config.get('min_confidence_threshold', 'Possible'),
            'max_abnormalities': self.symptom_config.get('max_abnormalities', 8),
            'enable_trend_analysis': self.symptom_config.get('enable_trend_analysis', True),
            'enable_statistical_analysis': self.symptom_config.get('enable_statistical_analysis', True)
        }
        
        # Initialize disease prediction components
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Enhanced disease priority categories
        self.high_priority_diseases = {
            'cancer', 'carcinoma', 'tumor', 'neoplasm', 'malignancy',
            'dementia', 'alzheimer', 'parkinson', 'huntington', 'lewy body',
            'frontotemporal', 'vascular dementia', 'cognitive impairment',
            'neurodegenerative', 'mild cognitive impairment', 'mci',
            'depression', 'bipolar', 'schizophrenia', 'psychosis'
        }
        
        self.moderate_priority_diseases = {
            'sleep apnea', 'osahs', 'insomnia', 'circadian rhythm',
            'irritable bowel syndrome', 'ibs', 'gastroenteritis',
            'hemorrhoids', 'fissure', 'polyp',
            'diabetes', 'insulinoma', 'hypoglycemia', 'thyroid'
        }
        
        # Symptom-to-condition mapping for better targeting
        self.symptom_condition_hints = {
            'sleep': ['dementia', 'sleep apnea', 'depression', 'anxiety'],
            'bathroom': ['urinary tract infection', 'diabetes', 'prostate', 'ibs'],
            'eating': ['dementia', 'depression', 'diabetes', 'insulinoma'],
            'isolation': ['dementia', 'depression', 'cognitive impairment'],
            'checking': ['ocd', 'dementia', 'anxiety', 'obsessive compulsive'],
            'repetitive': ['dementia', 'ocd', 'parkinson', 'autism'],
            'cognitive': ['dementia', 'alzheimer', 'mild cognitive impairment'],
            'memory': ['dementia', 'alzheimer', 'cognitive impairment']
        }
        
        logger.info("Integrated Medical Analysis Pipeline initialized")

    def summarize_locations(self, input_data):
        """Process raw location data into structured summary"""
        summary = []
        
        for date, entries in input_data.items():
            location_summary = {}
            for entry in entries:
                location = entry['location']
                if location not in location_summary:
                    location_summary[location] = {
                        'count': 0,
                        'entries': []
                    }
                
                location_summary[location]['entries'].append({
                    'start_time': entry['start'].split()[1],
                    'end_time': entry['end'].split()[1],
                    'duration': entry['duration_mins']
                })
                location_summary[location]['count'] += 1
            
            summary.append({
                'date': date,
                'locations': location_summary
            })
        
        return summary

    def preprocess_data(self, raw_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Preprocess the raw activity data to extract meaningful statistics and patterns"""
        try:
            if not raw_data:
                raise ValueError("Empty raw data provided")
            
            preprocessed = {
                'raw_data': raw_data,
                'date_range': {
                    'start': raw_data[0]['date'] if raw_data else None,
                    'end': raw_data[-1]['date'] if raw_data else None,
                    'total_days': len(raw_data)
                },
                'location_summary': {},
                'daily_patterns': {},
                'anomaly_flags': []
            }
            
            # Extract location statistics
            all_locations = set()
            for day in raw_data:
                all_locations.update(day.get('locations', {}).keys())
            
            for location in all_locations:
                preprocessed['location_summary'][location] = {
                    'total_entries': 0,
                    'total_duration': 0,
                    'avg_duration_per_entry': 0,
                    'days_active': 0,
                    'peak_hours': {},
                    'unusual_patterns': []
                }
            
            # Analyze each day
            for day_data in raw_data:
                date = day_data['date']
                locations = day_data.get('locations', {})
                
                daily_stats = {
                    'first_activity': None,
                    'last_activity': None,
                    'total_active_time': 0,
                    'location_distribution': {},
                    'activity_gaps': [],
                    'peak_activity_period': None
                }
                
                all_times = []
                
                for location, data in locations.items():
                    entries = data.get('entries', [])
                    count = data.get('count', len(entries))
                    total_duration = sum(entry.get('duration', 0) for entry in entries)
                    
                    # Update location summary
                    preprocessed['location_summary'][location]['total_entries'] += count
                    preprocessed['location_summary'][location]['total_duration'] += total_duration
                    preprocessed['location_summary'][location]['days_active'] += 1
                    
                    # Track times for daily analysis
                    for entry in entries:
                        start_time = entry.get('start_time', '')
                        end_time = entry.get('end_time', '')
                        all_times.extend([start_time, end_time])
                    
                    daily_stats['location_distribution'][location] = {
                        'count': count,
                        'total_duration': total_duration,
                        'avg_duration': total_duration / count if count > 0 else 0
                    }
                
                # Calculate daily patterns
                if all_times:
                    valid_times = [t for t in all_times if t and ':' in t]
                    if valid_times:
                        daily_stats['first_activity'] = min(valid_times)
                        daily_stats['last_activity'] = max(valid_times)
                
                daily_stats['total_active_time'] = sum(
                    loc_data['total_duration'] for loc_data in daily_stats['location_distribution'].values()
                )
                
                preprocessed['daily_patterns'][date] = daily_stats
            
            # Calculate averages for location summary
            for location, stats in preprocessed['location_summary'].items():
                if stats['total_entries'] > 0:
                    stats['avg_duration_per_entry'] = stats['total_duration'] / stats['total_entries']
                    stats['avg_entries_per_day'] = stats['total_entries'] / preprocessed['date_range']['total_days']
            
            logger.info(f"Preprocessed data for {len(raw_data)} days across {len(all_locations)} locations")
            return preprocessed
            
        except Exception as e:
            logger.error(f"Error in preprocessing data: {str(e)}")
            return {'raw_data': raw_data, 'preprocessing_error': str(e)}

    def detect_immediate_anomalies(self, preprocessed_data: Dict[str, Any]) -> List[str]:
        """Detect obvious anomalies before LLM analysis"""
        anomalies = []
        
        try:
            location_summary = preprocessed_data.get('location_summary', {})
            daily_patterns = preprocessed_data.get('daily_patterns', {})
            
            # Check for excessive activity in specific locations
            for location, stats in location_summary.items():
                avg_entries = stats.get('avg_entries_per_day', 0)
                if 'fridge' in location.lower() and avg_entries > 15:
                    anomalies.append(f"Excessive {location} activity: {avg_entries:.1f} entries per day")
                elif 'kitchen' in location.lower() and avg_entries > 60:
                    anomalies.append(f"Excessive {location} activity: {avg_entries:.1f} entries per day")
            
            # Check for irregular sleep patterns
            early_activities = 0
            late_activities = 0
            
            for date, patterns in daily_patterns.items():
                first_activity = patterns.get('first_activity', '')
                last_activity = patterns.get('last_activity', '')
                
                if first_activity and ':' in first_activity:
                    hour = int(first_activity.split(':')[0])
                    if hour < 5:
                        early_activities += 1
                
                if last_activity and ':' in last_activity:
                    hour = int(last_activity.split(':')[0])
                    if hour >= 23 or hour < 2:
                        late_activities += 1
            
            total_days = len(daily_patterns)
            if early_activities > total_days * 0.3:
                anomalies.append(f"Frequent very early activity: {early_activities}/{total_days} days")
            if late_activities > total_days * 0.5:
                anomalies.append(f"Frequent late night activity: {late_activities}/{total_days} days")
            
            logger.info(f"Detected {len(anomalies)} immediate anomalies")
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in immediate anomaly detection: {str(e)}")
            return []

    def _make_api_request(self, prompt: str, model: str = None, max_retries: int = 3) -> Optional[str]:
        """Make direct API request to OpenRouter with retry logic"""
        if model is None:
            model = self.symptom_model
            
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system" if model == self.disease_model else "user",
                    "content": "You are a medical expert specializing in differential diagnosis. Provide accurate, evidence-based medical analysis. Consider neurological and psychiatric conditions alongside organic diseases. Respond only in valid JSON format as specified, with no additional text, explanations, or markdown outside the JSON." if model == self.disease_model else prompt
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        if model == self.disease_model:
            payload["messages"].append({"role": "user", "content": prompt})
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Making API request (attempt {attempt + 1}/{max_retries})")
                response = requests.post(
                    self.openrouter_api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
                    if content and len(content.strip()) > 0:
                        logger.info("API request successful")
                        return content.strip()
                    else:
                        logger.warning(f"Empty content in API response on attempt {attempt + 1}")
                else:
                    logger.warning(f"API request failed on attempt {attempt + 1}: {response.status_code} - {response.text}")
                
            except requests.exceptions.Timeout:
                logger.warning(f"API request timed out on attempt {attempt + 1}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed on attempt {attempt + 1}: {str(e)}")
            except Exception as e:
                logger.warning(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
            
            if attempt < max_retries - 1:
                logger.info("Retrying in 2 seconds...")
                import time
                time.sleep(2)
            else:
                logger.error("Max retries exceeded")
        
        return None

    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response with multiple fallback strategies"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                # Try to extract JSON from response
                start = response.find("{")
                end = response.rfind("}") + 1
                
                if start != -1 and end > start:
                    json_str = response[start:end]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                pass
            
            try:
                # Try to find JSON within code blocks
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                
                # Try to find any JSON-like structure
                json_match = re.search(r'(\{[^{}]*"Abnormal Activity"[^{}]*\})', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
            except:
                pass
            
            logger.error(f"Failed to parse JSON from LLM response: {response[:200]}...")
            return None

    def detect_symptoms(self, preprocessed_summary_data):
        """Detect symptoms using the raw2ab logic"""
        try:
            # Preprocess data for enhanced analysis
            preprocessed_data = self.preprocess_data(preprocessed_summary_data)
            
            # Detect immediate anomalies
            immediate_anomalies = self.detect_immediate_anomalies(preprocessed_data)
            
            # Prepare enhanced context for LLM
            analysis_context = {
                'preprocessed_data': preprocessed_data,
                'immediate_anomalies': immediate_anomalies,
                'analysis_config': self.analysis_config
            }
            
            # Format prompt with enhanced data
            formatted_prompt = enhanced_prompt_template.format(
                raw_data=json.dumps(preprocessed_summary_data, indent=2),
                analysis_context=json.dumps(analysis_context, indent=2, default=str)
            )
            
            logger.info("Sending request to OpenRouter API for symptom analysis...")
            
            # Get LLM response
            response = self._make_api_request(formatted_prompt)
            
            if not response:
                raise Exception("Failed to get response from OpenRouter API after retries")
            
            # Parse and validate response
            output_json = self._parse_llm_response(response)
            
            if not output_json:
                raise Exception("Failed to parse valid JSON from LLM response")
            
            logger.info(f"Detected {len(output_json.get('Abnormal Activity', []))} abnormal activities")
            return output_json.get('Abnormal Activity', [])
            
        except Exception as e:
            logger.error(f"Error in symptom detection: {str(e)}")
            return []

    def get_disease_priority(self, disease_name):
        """Assign priority score to diseases based on medical importance"""
        disease_lower = disease_name.lower()
        
        for high_priority in self.high_priority_diseases:
            if high_priority in disease_lower:
                return 3
        
        for moderate_priority in self.moderate_priority_diseases:
            if moderate_priority in disease_lower:
                return 2
                
        return 1

    def generate_targeted_queries(self, symptom_data):
        """Generate targeted queries based on symptom content and condition hints"""
        symptom_text = symptom_data.get('Abnormal Activity', '')
        definition = symptom_data.get('definition', '')
        
        base_queries = [
            f"Disease diagnosis symptoms: {symptom_text}",
            f"Medical condition causing: {symptom_text}",
            f"Differential diagnosis for: {symptom_text}",
            f"Clinical presentation: {symptom_text}",
            f"Medical condition: {definition}"
        ]
        
        # Add targeted queries based on symptom content
        symptom_lower = (symptom_text + " " + definition).lower()
        targeted_queries = []
        
        for keyword, conditions in self.symptom_condition_hints.items():
            if keyword in symptom_lower:
                for condition in conditions:
                    targeted_queries.extend([
                        f"{condition} symptoms {symptom_text}",
                        f"{condition} diagnosis {keyword}",
                        f"{condition} clinical features"
                    ])
        
        return base_queries + targeted_queries

    def enhanced_symptom_retrieval(self, symptom_data, retriever):
        """Enhanced retrieval with multiple query strategies"""
        queries = self.generate_targeted_queries(symptom_data)
        
        all_docs = []
        seen_content = set()
        
        for query in queries:
            try:
                retrieved_docs = retriever.invoke(query)
                for doc in retrieved_docs:
                    content_hash = hash(doc.page_content[:500])
                    if content_hash not in seen_content:
                        all_docs.append(doc)
                        seen_content.add(content_hash)
            except Exception as e:
                print(f"Warning: Retriever failed for query '{query}': {str(e)}")
                continue
        
        # Rerank all documents
        if all_docs:
            symptom_text = symptom_data.get('Abnormal Activity', '')
            pairs = [[f"Medical diagnosis for symptom: {symptom_text}", doc.page_content] for doc in all_docs]
            scores = self.reranker.predict(pairs)
            
            # Boost scores for high-priority conditions
            adjusted_scores = []
            for i, (score, doc) in enumerate(zip(scores, all_docs)):
                content_lower = doc.page_content.lower()
                boost = 0
                
                for disease in self.high_priority_diseases:
                    if disease in content_lower:
                        boost += 0.1
                
                adjusted_scores.append(score + boost)
            
            sorted_indices = np.argsort(adjusted_scores)[::-1][:25]
            return [all_docs[i] for i in sorted_indices]
        
        return []

    def extract_disease_mentions(self, context):
        """Enhanced disease extraction with focus on neurological conditions"""
        disease_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:cancer|carcinoma|tumor|disease|syndrome|disorder|dementia)\b',
            r'\b(?:cancer|carcinoma|tumor|disease|syndrome|disorder|dementia)\s+of\s+([a-z\s]+)\b',
            r'\b(alzheimer\'?s?\s+disease|dementia|vascular\s+dementia|lewy\s+body\s+dementia)\b',
            r'\b(frontotemporal\s+dementia|parkinson\'?s?\s+disease|huntington\'?s?\s+disease)\b',
            r'\b(mild\s+cognitive\s+impairment|MCI|cognitive\s+impairment)\b',
            r'\b(major\s+depressive\s+disorder|depression|bipolar\s+disorder|schizophrenia)\b',
            r'\b(obsessive[-\s]compulsive\s+disorder|OCD|anxiety\s+disorder)\b',
            r'\b(inflammatory\s+bowel\s+disease|IBD|ulcerative\s+colitis|crohn\'?s?\s+disease)\b',
            r'\b(sleep\s+apnea|OSAHS|insomnia|diabetes|insulinoma)\b'
        ]
        
        diseases = set()
        for pattern in disease_patterns:
            matches = re.finditer(pattern, context, re.IGNORECASE)
            for match in matches:
                disease = match.group(1) if match.group(1) else match.group(0)
                diseases.add(disease.strip().title())
        
        return list(diseases)

    def predict_diseases(self, symptom_data_list):
        """Predict diseases based on enhanced symptom data"""
        print("Loading RAG database...")
        
        # Load RAG configuration and embeddings
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embeddings = AutoEmbedding(
            self.rag_config["model_name"], 
            self.rag_config["embedding_type"], 
            model_kwargs={"device": device}
        )
        
        try:
            rag_db = FAISS.load_local(
                self.rag_config["faiss_path"], 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            raise Exception(f"Failed to load FAISS database: {str(e)}")

        # Set up retriever
        retriever = rag_db.as_retriever(search_kwargs={"k": 60})

        print("Processing enhanced symptoms and retrieving medical evidence...")
        
        # Enhanced per-symptom data collection
        per_symptom_data = {}
        disease_mentions = defaultdict(list)
        
        # Create symptom list for prompt
        enhanced_symptoms_for_prompt = []
        for i, symptom_data in enumerate(symptom_data_list):
            symptom_text = symptom_data.get('Abnormal Activity', '')
            confidence = symptom_data.get('confidence', '')
            definition = symptom_data.get('definition', '')
            
            enhanced_symptom_description = f"{i+1}. {symptom_text} (Confidence: {confidence}, Definition: {definition})"
            enhanced_symptoms_for_prompt.append(enhanced_symptom_description)

        for i, symptom_data in enumerate(symptom_data_list):
            symptom_text = symptom_data.get('Abnormal Activity', '')
            print(f"Processing symptom {i+1}/{len(symptom_data_list)}: {symptom_text[:50]}...")
            
            # Enhanced retrieval using all symptom information
            top_docs = self.enhanced_symptom_retrieval(symptom_data, retriever)
            
            if not top_docs:
                print(f"Warning: No documents retrieved for symptom: {symptom_text}")
                continue
            
            # Build enhanced context
            context = f"=== MEDICAL EVIDENCE FOR ENHANCED SYMPTOM: {symptom_text} ===\n"
            context += f"Confidence Level: {symptom_data.get('confidence', 'Unknown')}\n"
            context += f"Clinical Definition: {symptom_data.get('definition', 'Not provided')}\n\n"
            
            for j, doc in enumerate(top_docs):
                context += f"--- Medical Reference {j+1} ---\n"
                context += doc.page_content.strip()
                context += f"\n\n"
                
                # Extract disease mentions
                diseases_in_doc = self.extract_disease_mentions(doc.page_content)
                for disease in diseases_in_doc:
                    disease_mentions[disease].append(symptom_text)

            per_symptom_data[symptom_text] = context

        # Create consolidated context
        consolidated_context = ""
        for symptom, context in per_symptom_data.items():
            consolidated_context += f"\n\n{context}"

        print("Generating enhanced disease predictions...")
        
        # Enhanced prompt with specific symptom data integration
        prompt = f"""
You are conducting a comprehensive medical differential diagnosis analysis using ENHANCED SYMPTOM DATA that includes abnormal activities, confidence levels, and clinical definitions.

CRITICAL CONSIDERATIONS:
- Each symptom comes with CONFIDENCE LEVEL and CLINICAL DEFINITION - use these to weight your analysis
- Higher confidence symptoms should have greater influence on disease predictions
- Clinical definitions provide important context for medical interpretation
- Pay special attention to NEUROLOGICAL or PSYCHIATRIC conditions
- Consider how behavioral changes, sleep disruption, and repetitive behaviors relate to cognitive decline

ENHANCED PATIENT SYMPTOMS TO ANALYZE:
{chr(10).join(enhanced_symptoms_for_prompt)}

ENHANCED ANALYSIS REQUIREMENTS:
1. CONFIDENCE WEIGHTING: Give more weight to "Very Likely" and "Likely" symptoms in your analysis
2. DEFINITION INTEGRATION: Use the clinical definitions to better understand symptom significance
3. COMPREHENSIVE COVERAGE: Ensure the selected diseases collectively explain ALL patient symptoms
4. NEUROLOGICAL FOCUS: Specifically evaluate for dementia, Alzheimer's disease, and cognitive impairment
5. EVIDENCE-BASED: Base conclusions strictly on the provided medical contexts
6. PRIORITIZATION: Include serious conditions even if less likely

CONFIDENCE LEVELS FOR DISEASE PREDICTION:
- Very Likely (80-95%): Strong evidence, multiple high-confidence symptoms match
- Likely (50-79%): Good evidence, some symptoms match well with clinical definitions
- Possible (40-49%): Some evidence, definitions support possibility
- Unlikely (20-39%): Weak evidence, minimal symptom-definition match
- Very Unlikely (<20%): Very weak evidence, but cannot be ruled out

MEDICAL EVIDENCE AND CONTEXTS:
{consolidated_context}

For each disease you identify, provide:
- disease: Exact medical name from the contexts
- related_abnormal_activities: List ALL patient symptoms this disease could explain (use original symptom text)
- reasoning: Detailed medical reasoning including:
  * How confidence levels of matching symptoms support this diagnosis
  * How clinical definitions align with this condition
  * Evidence from medical contexts supporting this diagnosis
  * Clinical significance and progression patterns
- confidence: Choose from defined levels based on strength of evidence and symptom-definition alignment

Return ONLY valid JSON in this exact format:
{{"disease_predictions": [{{"disease": "Disease Name", "related_abnormal_activities": ["symptom1", "symptom2"], "reasoning": "detailed explanation", "confidence": "Confidence Level"}}, ...]}}
"""

        # Get LLM prediction
        try:
            llm_response = self._make_api_request(prompt, self.disease_model)
            # Clean response
            cleaned_response = re.sub(r'^```json\s*|\s*```$', '', llm_response, flags=re.MULTILINE).strip()
            prediction_result = json.loads(cleaned_response)
            
            # Post-process results
            if "disease_predictions" in prediction_result:
                diseases = prediction_result["disease_predictions"]
                
                # Enhanced sorting with confidence and definition weighting
                def sort_key(disease):
                    confidence_scores = {
                        "Very Likely": 5, "Likely": 4, "Possible": 3, 
                        "Unlikely": 2, "Very Unlikely": 1
                    }
                    priority = self.get_disease_priority(disease.get("disease", ""))
                    confidence = confidence_scores.get(disease.get("confidence", "Unlikely"), 2)
                    
                    # Extra boost for neurological conditions
                    disease_name = disease.get("disease", "").lower()
                    neuro_boost = 0
                    if any(term in disease_name for term in ['dementia', 'alzheimer', 'cognitive', 'parkinson']):
                        neuro_boost = 5
                    
                    return (priority * 10 + confidence + neuro_boost)
                
                diseases.sort(key=sort_key, reverse=True)
                prediction_result["disease_predictions"] = diseases
            
            return prediction_result
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Raw response: {llm_response}")
            raise ValueError(f"Invalid JSON response from LLM: {llm_response}")

    def run_complete_analysis(self, raw_input_data):
        """
        Run the complete integrated analysis pipeline
        """
        try:
            print("="*60)
            print("STARTING INTEGRATED MEDICAL ANALYSIS PIPELINE")
            print("="*60)
            
            # STEP 1: Convert to summary format (summary.py logic)
            print("\nSTEP 1: Processing raw data...")
            if isinstance(raw_input_data, dict):
                summary_data = self.summarize_locations(raw_input_data)
            else:
                summary_data = raw_input_data  # Assume already in summary format
            
            print(f"Processed data for {len(summary_data)} days")
            
            # STEP 2: Detect symptoms
            print("\nSTEP 2: Detecting abnormal activity patterns...")
            symptom_data_list = self.detect_symptoms(summary_data)
            
            if not symptom_data_list:
                print("No significant abnormal patterns detected.")
                return {
                    "analysis_metadata": {
                        "analysis_date": datetime.now().isoformat(),
                        "total_symptoms": 0,
                        "total_diseases": 0,
                        "status": "No abnormalities detected"
                    },
                    "enhanced_symptoms": [],
                    "disease_predictions": []
                }
            
            print(f"Detected {len(symptom_data_list)} abnormal activity patterns")
            
            # STEP 3: Predict diseases based on enhanced symptom data
            print("\nSTEP 3: Predicting diseases from enhanced symptoms...")
            disease_predictions = self.predict_diseases(symptom_data_list)
            
            print(f"Generated {len(disease_predictions.get('disease_predictions', []))} disease predictions")
            
            # STEP 4: Compile final enhanced output
            print("\nSTEP 4: Compiling enhanced analysis results...")
            
            enhanced_output = {
                "analysis_metadata": {
                    "analysis_date": datetime.now().isoformat(),
                    "data_date_range": {
                        "start": summary_data[0]['date'] if summary_data else None,
                        "end": summary_data[-1]['date'] if summary_data else None,
                        "total_days": len(summary_data)
                    },
                    "models_used": {
                        "symptom_detection": self.symptom_model,
                        "disease_prediction": self.disease_model
                    },
                    "total_symptoms": len(symptom_data_list),
                    "total_diseases": len(disease_predictions.get('disease_predictions', [])),
                    "confidence_threshold": self.analysis_config['min_confidence_threshold']
                },
                "enhanced_symptoms": symptom_data_list,
                **disease_predictions
            }
            
            return enhanced_output
            
        except Exception as e:
            logger.error(f"Error in complete analysis pipeline: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


# Configuration for the integrated pipeline
config = {
    "symptom_detection": {
        'api_key': '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',
        'model': 'deepseek/deepseek-chat-v3.1',
        'temperature': 0.1,
        'max_tokens': 10000,
        'min_confidence_threshold': 'Possible',
        'max_abnormalities': 8,
        'enable_trend_analysis': True,
        'enable_statistical_analysis': True
    },
    "disease_prediction": {
        'model': "google/gemini-2.5-pro"
    },
    "rag_db": {
        "faiss_path": "./rag",
        "docs_path": "./ocr/output", 
        "model_name": "pritamdeka/S-PubMedBert-MS-MARCO",
        "embedding_type": "sentence_transformer",
        "chunk_size": 1024,
        "chunk_overlap": 256
    }
}

# Initialize the pipeline
pipeline = IntegratedMedicalAnalysisPipeline(config)

@app.get("/")
async def root():
    return {"message": "Medical Analysis API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/analyze")
async def analyze_medical_data(file: UploadFile = File(...)):
    """
    Analyze medical data from uploaded JSON file
    """
    try:
        # Validate file type
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="File must be a JSON file")
        
        # Read and parse JSON data
        content = await file.read()
        try:
            data = json.loads(content.decode('utf-8'))
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file")
        
        # Run analysis
        logger.info(f"Starting analysis for file: {file.filename}")
        results = pipeline.run_complete_analysis(data)
        
        if results is None:
            raise HTTPException(status_code=500, detail="Analysis failed")
        
        logger.info("Analysis completed successfully")
        return JSONResponse(content=results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Medical Analysis API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
