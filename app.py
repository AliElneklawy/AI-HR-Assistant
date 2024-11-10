import gradio as gr
import pandas as pd
from datetime import datetime
import json
from typing import Tuple, Dict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from en_employees_chatbot import HRChatbot
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dotenv import load_dotenv



class HRChatbotGradio(HRChatbot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def chat_interface(self, message: str, history: list) -> str:
        """Wrapper for get_response to work with Gradio chat interface"""
        response = self.get_response(message)
        return response
    
    def process_leave_request(self, 
                            employee_id: str,
                            leave_type: str,
                            start_date: str,
                            end_date: str) -> str:
        """Wrapper for create_leave_request to work with Gradio interface"""
        try:
            employee_data = {
                'employee_id': employee_id,
                'leave_type': leave_type,
                'start_date': start_date,
                'end_date': end_date,
                'status': 'pending'
            }
            
            result = self.create_leave_request(employee_data)
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error processing request: {str(e)}"
    
    def get_stats_interface(self) -> Tuple[str, gr.Plot]:
        """Wrapper for get_statistics with visualization"""
        try:
            stats = self.get_statistics()
            
            if os.path.exists(self.leaves_file):
                df = pd.read_excel(self.leaves_file)
                
                # Create subplot figure
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Leave Types Distribution', 
                                'Request Status Distribution',
                                'Leave Requests Over Time'),
                    specs=[[{'type': 'bar'}, {'type': 'pie'}],
                        [{'type': 'scatter', 'colspan': 2}, None]]
                )
                
                # 1. Leave Types Distribution (Bar Chart)
                leave_counts = df['leave_type'].value_counts()
                fig.add_trace(
                    go.Bar(
                        x=leave_counts.index,
                        y=leave_counts.values,
                        name='Leave Types',
                        marker_color='rgb(55, 83, 109)'
                    ),
                    row=1, col=1
                )
                
                # 2. Status Distribution (Pie Chart)
                status_counts = df['status'].value_counts()
                fig.add_trace(
                    go.Pie(
                        labels=status_counts.index,
                        values=status_counts.values,
                        hole=.3,
                        name='Status'
                    ),
                    row=1, col=2
                )
                
                # 3. Time-based analysis
                if len(df) > 0:
                    df['start_date'] = pd.to_datetime(df['start_date'])
                    df['end_date'] = pd.to_datetime(df['end_date'])
                    df['duration'] = (df['end_date'] - df['start_date']).dt.days
                    
                    # Monthly trend
                    monthly_counts = df.groupby(df['start_date'].dt.strftime('%Y-%m')).size()
                    fig.add_trace(
                        go.Scatter(
                            x=monthly_counts.index,
                            y=monthly_counts.values,
                            mode='lines+markers',
                            name='Monthly Requests'
                        ),
                        row=2, col=1
                    )
                
                # Update layout
                fig.update_layout(
                    height=800,
                    showlegend=True,
                    title_text="Leave Statistics Dashboard",
                )
                
                # Update axes labels
                fig.update_xaxes(title_text="Leave Type", row=1, col=1)
                fig.update_yaxes(title_text="Number of Requests", row=1, col=1)
                fig.update_xaxes(title_text="Month", row=2, col=1)
                fig.update_yaxes(title_text="Number of Requests", row=2, col=1)
                
                return json.dumps(stats, indent=2, ensure_ascii=False), fig
            else:
                # Create empty figure with message
                empty_fig = go.Figure()
                empty_fig.add_annotation(
                    text="No leave data available yet",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False
                )
                return "No leave data available yet", empty_fig
        except Exception as e:
            print(f"Detailed error: {str(e)}")  # For debugging
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text=f"Error generating statistics: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )
            return f"Error generating statistics: {str(e)}", empty_fig

def create_gradio_interface():

    load_dotenv()
    
    chatbot = HRChatbotGradio(
        palm_api_key=os.getenv('PALM_API_KEY'),
        training_data_dir='company_data',
        leaves_file='leaves.xlsx'
    )
    
    # Create the interface
    with gr.Blocks(title="HR Assistant", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🤖 HR Assistant")
        
        with gr.Tabs():
            # Chat Tab
            with gr.Tab("Chat with HR Assistant"):
                chatbot_interface = gr.ChatInterface(
                    chatbot.chat_interface,
                    examples=[
                        "What are the leave policies?",
                        "How many sick days am I entitled to?",
                        "What are the working hours?",
                    ],
                    title="Chat with HR Assistant",
                )
            
            # Leave Request Tab
            with gr.Tab("Submit Leave Request"):
                with gr.Column():
                    gr.Markdown("### Submit Leave Request")
                    employee_id = gr.Textbox(label="Employee ID")
                    leave_type = gr.Dropdown(
                        choices=["Sick Leave", "Annual Leave", "Personal Leave", "Other"],
                        label="Leave Type"
                    )
                    start_date = gr.Textbox(
                        label="Start Date (YYYY-MM-DD)",
                        placeholder="2024-01-01"
                    )
                    end_date = gr.Textbox(
                        label="End Date (YYYY-MM-DD)",
                        placeholder="2024-01-02"
                    )
                    submit_btn = gr.Button("Submit Request")
                    result_text = gr.Textbox(
                        label="Result",
                        interactive=False
                    )
                    
                    submit_btn.click(
                        fn=chatbot.process_leave_request,
                        inputs=[employee_id, leave_type, start_date, end_date],
                        outputs=result_text
                    )
            
            # Statistics Tab
            with gr.Tab("Statistics"):
                with gr.Column():
                    gr.Markdown("### Leave Statistics and Insights")
                    refresh_btn = gr.Button("Refresh Statistics")
                    stats_text = gr.Textbox(
                        label="Statistics",
                        interactive=False
                    )
                    stats_plot = gr.Plot(label="Leave Statistics Dashboard")
                    
                    refresh_btn.click(
                        fn=chatbot.get_stats_interface,
                        inputs=[],
                        outputs=[stats_text, stats_plot]
                    )
        
        # Footer
        gr.Markdown("### 📝 Documentation")
        with gr.Accordion("How to use"):
            gr.Markdown("""
            1. **Chat Tab**: Ask questions about HR policies and procedures
            2. **Leave Request Tab**: Submit new leave requests
            3. **Statistics Tab**: View leave statistics and insights
            
            For support, contact HR department.
            """)
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_gradio_interface()
    demo.launch(share=True)