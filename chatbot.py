import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import pandas as pd
from datetime import datetime
import json
import os
from collections import Counter
import PyPDF2
import os
import stat
from dotenv import load_dotenv

    
class HRChatbot:
    def __init__(self, palm_api_key: str, training_data_dir: str = "company_data", leaves_file: str = "leaves.xlsx"):
        """
        Initialize the chatbot with PaLM API and local embeddings
        """
        # Setup PaLM
        print('PaLm setup')
        genai.configure(api_key=palm_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Setup local embeddings model
        print('Embedding model setup')
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Initialize storage
        self.leaves_file = leaves_file
        self.chat_history = []
        self.query_history = []
        
        # Load and embed company knowledge base
        print('Loading knowledge base')
        self.knowledge_base = self._load_knowledge_base(training_data_dir)

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from PDF files
        """
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {str(e)}")
            return ""
              
    def _load_knowledge_base(self, training_data_dir: str) -> Dict[str, np.ndarray]:
        """
        Load and embed company documents (both txt and pdf files)
        """
        knowledge_base = {}
        
        if not os.path.exists(training_data_dir):
            os.makedirs(training_data_dir)
            # Create sample files if directory is empty
            sample_data = {
                'policies.txt': 'Leave Policies:\n- Annual leave 30 days\n- Sick leave 15 days',
                'rules.txt': 'Work Rules:\n- Working hours from 8 AM to 4 PM',
                'faq.txt': 'Frequently Asked Questions:\n- How do I apply for leave?\n- What is the flexible work policy?'
            }
            for filename, content in sample_data.items():
                with open(os.path.join(training_data_dir, filename), 'w', encoding='utf-8') as f:
                    f.write(content)
        
        # Process all txt and pdf files in the directory
        for filename in os.listdir(training_data_dir):
            file_path = os.path.join(training_data_dir, filename)
            content = ""
            
            if filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
            elif filename.endswith('.pdf'):
                content = self._extract_text_from_pdf(file_path)
            else:
                continue
                
            if content.strip():  # Only process non-empty content
                # Split content into chunks
                chunks = self._split_text(content)
                # Create embeddings for each chunk
                embeddings = self.embedding_model.encode(chunks)
                knowledge_base[filename] = {
                    'chunks': chunks,
                    'embeddings': embeddings
                }
                print(f"Processed {filename}: {len(chunks)} chunks created")
            else:
                print(f"Warning: No content extracted from {filename}")
        
        return knowledge_base
    
    def _split_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Split text into smaller chunks, handling PDF-specific formatting
        """
        # Clean up common PDF artifacts
        text = text.replace('\n\n', ' ').replace('  ', ' ').strip()
        
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_length += len(word) + 1  # +1 for space
            if current_length > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def _find_relevant_context(self, query: str, top_k: int = 3) -> str:
        """
        Find most relevant context from knowledge base using embeddings
        """
        query_embedding = self.embedding_model.encode([query])[0]
        relevant_chunks = []
        
        for doc in self.knowledge_base.values():
            similarities = np.dot(doc['embeddings'], query_embedding)
            top_indices = np.argsort(similarities)[-top_k:]
            
            for idx in top_indices:
                relevant_chunks.append(doc['chunks'][idx])
        
        return "\n".join(relevant_chunks)

    def get_response(self, query: str) -> str:
        """
        Get response from PaLM with relevant context
        """
        self.query_history.append(query)
        
        # Get relevant context
        context = self._find_relevant_context(query)
        
        # Create prompt with context
        prompt = f"""
        As an HR assistant, use the following context to answer the question.
        If the answer is not in the context, use your general knowledge and mention that.
        
        Context:
        {context}
        
        Previous conversation:
        {self.chat_history[-3:] if self.chat_history else 'No previous conversation'}
        
        Question: {query}
        """
        
        # Get response from PaLM
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
        except Exception as e:
            response_text = f"Sorry, I couldn't process this request. Error: {str(e)}"
        
        # Update chat history
        self.chat_history.append({"question": query, "answer": response_text})
        
        return response_text

    def create_leave_request(self, employee_data: Dict) -> Dict:
        """
        Create and validate leave request
        """
        # Validate request using PaLM
        validation_prompt = f"""
        Validate the following leave request based on common HR policies:
        Employee ID: {employee_data['employee_id']}
        Leave Type: {employee_data['leave_type']}
        Start Date: {employee_data['start_date']}
        End Date: {employee_data['end_date']}
        
        Check for:
        1. Reasonable leave duration.
        2. Valid leave type
        
        Reply with 'valid' or 'invalid: <reason>'
        """
        
        validation = self.model.generate_content(validation_prompt)
        
        if "invalid" in validation.text.lower():
            return {"status": "rejected", "reason": validation.text}
            
        try:
            # Create leaves.xlsx if it doesn't exist
            if not os.path.exists(self.leaves_file):
                df = pd.DataFrame(columns=['employee_id', 'leave_type', 'start_date', 'end_date', 'status'])
                df.to_excel(self.leaves_file, index=False)
            
            # Save to Excel
            df = pd.read_excel(self.leaves_file)
            df = pd.concat([df, pd.DataFrame([employee_data])], ignore_index=True)
            df.to_excel(self.leaves_file, index=False)
            
            return {"status": "success", "message": "Leave request created successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_statistics(self) -> Dict:
        """
        Generate statistics and insights using PaLM
        """
        try:
            df = pd.read_excel(self.leaves_file)
            stats = {
                "total_requests": len(df),
                "leave_types": df['leave_type'].value_counts().to_dict(),
                "status_distribution": df['status'].value_counts().to_dict(),
                "query_categories": Counter(self.query_history)
            }
            
            # Get insights from PaLM
            stats_prompt = f"""
            Analyze the following HR statistics and provide insights:
            {json.dumps(stats, indent=2, ensure_ascii=False)}
            
            Provide 3-4 key insights about:
            1. Leave request patterns
            2. Common queries
            3. Potential improvements
            """
            
            insights = self.model.generate_content(stats_prompt)
            
            return {
                "statistics": stats,
                "insights": insights.text
            }
            
        except Exception as e:
            return {"error": f"Error generating statistics: {str(e)}"}

def main():
    
    chatbot = HRChatbot(
        palm_api_key=os.getenv('PALM_API_KEY'),
        training_data_dir='company_data',
        leaves_file='leaves.xlsx'
    )

    if os.path.exists('leaves.xlsx'):
        # Give read/write permissions
        os.chmod('leaves.xlsx', stat.S_IRUSR | stat.S_IWUSR)
    
    while True:
        print("\nHR Assistant:")
        print("1. Ask a question")
        print("2. Apply for vacation")
        print("3. Statistics")
        print("4. Exit")
        
        choice = input("Choice: ")
        
        if choice == '1':
            question = input("Your question: ")
            response = chatbot.get_response(question)
            print(f"\nAnswer: {response}")
            
        elif choice == '2':
            employee_data = {
                'employee_id': input("Employee ID: "),
                'leave_type': input("Leave type: "),
                'start_date': input("Start date (YYYY-MM-DD): "),
                'end_date': input("End date (YYYY-MM-DD): "),
                'status': 'pending'
            }
            result = chatbot.create_leave_request(employee_data)
            print(f"\nResult: {result}")
            
        elif choice == '3':
            stats = chatbot.get_statistics()
            print("\nStatistics and Insights:")
            print(json.dumps(stats, indent=2, ensure_ascii=False))
            
        elif choice == '4':
            break
            
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main()