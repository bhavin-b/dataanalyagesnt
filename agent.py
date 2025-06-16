#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from together import Together
from docx import Document
from PyPDF2 import PdfReader
import io
import base64
from dotenv import load_dotenv

load_dotenv()

class DataAnalystAgent:
    def __init__(self, api_key: str):
        self.data = None
        self.text_content = None
        self.image_content = None
        self.analysis_history = []
        self.model = 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8'
        if not api_key:
            raise ValueError("TOGETHER_API_KEY not found in .env file or provided")
        self.client = Together(api_key=api_key)

    def load_document(self, file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext in ['.csv', '.xlsx']:
                self.data = pd.read_csv(file_path) if ext == '.csv' else pd.read_excel(file_path)
                return {'status': 'success', 'message': f'Data loaded successfully. Shape: {self.data.shape}'}
            elif ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.text_content = f.read()
                return {'status': 'success', 'message': 'Text file loaded successfully.'}
            elif ext == '.docx':
                doc = Document(file_path)
                self.text_content = '\n'.join([para.text for para in doc.paragraphs])
                return {'status': 'success', 'message': 'DOCX file loaded successfully.'}
            elif ext == '.pdf':
                reader = PdfReader(file_path)
                self.text_content = '\n'.join([page.extract_text() for page in reader.pages if page.extract_text()])
                return {'status': 'success', 'message': 'PDF file loaded successfully.'}
            elif ext in ['.png', '.jpg', '.jpeg']:
                with open(file_path, 'rb') as image_file:
                    self.image_content = base64.b64encode(image_file.read()).decode('utf-8')
                return {'status': 'success', 'message': 'Image file loaded successfully.'}
            else:
                return {'status': 'error', 'message': 'Unsupported file type.'}
        except Exception as e:
            return {'status': 'error', 'message': f'Error loading file: {str(e)}'}

    def get_data_summary(self):
        if self.data is not None:
            # For structured data (CSV/Excel)
            shape = self.data.shape
            missing_values = self.data.isnull().sum().to_dict()
            dtypes = self.data.dtypes.apply(lambda x: str(x)).to_dict()
            sample_data = self.data.head().to_dict(orient='records')
            
            return {
                'type': 'structured',
                'shape': shape,
                'missing_values': missing_values,
                'dtypes': dtypes,
                'sample_data': sample_data
            }
        elif self.text_content is not None:
            # For unstructured data (text files)
            return {
                'type': 'text',
                'length': len(self.text_content),
                'word_count': len(self.text_content.split()),
                'preview': self.text_content[:500]
            }
        elif self.image_content is not None:
            # For image files
            return {
                'type': 'image',
                'message': 'Image summary is not available. Ask questions about the image content.'
            }
        else:
            return {'error': 'No data loaded.'}

    def create_visualizations(self):
        if self.data is None:
            return [{'error': 'No data loaded for visualization.'}]

        visualizations = []
        numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()

        if not numeric_cols:
            return [{'error': 'No numeric columns to visualize.'}]

        # Create a histogram for the first numeric column
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            fig = px.histogram(self.data, x=col, title=f'Histogram of {col}')
            visualizations.append({
                'title': f'Histogram of {col}',
                'figure': fig,
                'insights': f"This histogram shows the distribution of {col}."
            })

        # Create a scatter plot for the first two numeric columns
        if len(numeric_cols) > 1:
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            fig = px.scatter(self.data, x=x_col, y=y_col, title=f'Scatter Plot: {x_col} vs {y_col}')
            visualizations.append({
                'title': f'Scatter Plot: {x_col} vs {y_col}',
                'figure': fig,
                'insights': f"This scatter plot shows the relationship between {x_col} and {y_col}."
            })
            
        return visualizations

    def answer_question(self, question: str, chat_history: list = None):
        context = ''
        if self.data is not None:
            context += f'Dataset Info: {self.data.head().to_string()}\nShape: {self.data.shape}\n'
        if self.text_content:
            context += f'Text Content: {self.text_content[:1000]}...\n'
        if self.image_content:
            context += 'Image content is available but cannot be displayed in text form.\n'

        messages = []
        if chat_history:
            for message in chat_history:
                messages.append({'role': message['role'], 'content': message['content']})
        
        messages.append({'role': 'user', 'content': f"Context: {context}\nQuestion: {question}"})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000
            )
            
            llm_answer = ""
            choices = getattr(response, 'choices', None)
            if choices and len(choices) > 0:
                message = getattr(choices[0], 'message', None)
                if message:
                    content = getattr(message, 'content', None)
                    if content:
                        llm_answer = content

            if not llm_answer:
                llm_answer = "Error: Could not parse LLM response."

            # For now, quantitative_answer is a placeholder.
            quantitative_answer = {'details': 'No specific quantitative data extracted.'}

            self.analysis_history.append({
                'question': question,
                'answer': llm_answer,
                'context': context
            })

            return {
                'llm_answer': llm_answer,
                'quantitative_answer': quantitative_answer
            }
        except Exception as e:
            return {'error': f'Error answering question: {str(e)}'}