import datetime
from streamlit_quill import st_quill
import streamlit as st
import time
import torch
import re
import requests
import json

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_session_state():
    if 'job_data' not in st.session_state:
        st.session_state.job_data = {
            'position': '',
            'display_name': '',
            'recruitment_type': 'External',
            'vacancies': 1,
            'start_date': datetime.date.today(),
            'target_date': datetime.date.today(),
            'country': '',
            'city': '',
            'work_type': '',
            'work_class': '',
            'business_type': '',
            'feedback': '',
            'language': 'English'  # Add default language
        }

if 'is_generated' not in st.session_state:
    st.session_state.is_generated = False
if 'english_description' not in st.session_state:
    st.session_state.english_description = ""
if 'arabic_description' not in st.session_state:
    st.session_state.arabic_description = ""
if 'generation_time' not in st.session_state:
    st.session_state.generation_time = 0
# Add new session state variables for feedback descriptions
if 'english_feedback_description' not in st.session_state:
    st.session_state.english_feedback_description = ""
if 'arabic_feedback_description' not in st.session_state:
    st.session_state.arabic_feedback_description = ""



# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configure the page
st.set_page_config(page_title="Job Posting System", layout="wide")

# Add custom CSS for styling
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] { height: 4rem; }
    .stRadio > label {
        background-color: #f0f2f6;
        padding: 10px 20px;
        border-radius: 5px;
        margin-right: 10px;
    }
    .quill { background-color: white; }
    </style>
""", unsafe_allow_html=True)




def generate_job_description(feedback=None):
    try:
        start_time = time.time()
        system_prompt = """You are an experienced HR professional crafting a detailed job description.
Create a clear and professional job description following this EXACT format.
Important: Generate ONLY the job description text, no other text or analysis and do not generate any text after the end in the generated job description or in the feedback section  .


Job Title: [title]

Location: [city], [country]

Work Type: [type]

Level: [level]

Industry: [industry]

Overview:
[2-3 sentences about the role and its impact]

Key Responsibilities:
• [Clear, action-oriented responsibility]
• [Clear, action-oriented responsibility]
• [Clear, action-oriented responsibility]
• [Clear, action-oriented responsibility]
• [Clear, action-oriented responsibility]

Required Qualifications:
• [Specific required qualification]
• [Specific required qualification]
• [Specific required qualification]
• [Specific required qualification]

Skills & Experience:
• [Specific skill or experience]
• [Specific skill or experience]
• [Specific skill or experience]
• [Specific skill or experience]

Additional Requirements:
• [Additional requirement]
• [Additional requirement]
• [Additional requirement]

Work Environment:
[Brief description of work environment and culture]"""

        # Format content
        city = st.session_state.job_data['city']
        country = st.session_state.job_data['country']
        content = f"""Create a professional job description for:
        Position: {st.session_state.job_data['position']}
        Location: {city}, {country}
        Work Type: {st.session_state.job_data['work_type']}
        Level: {st.session_state.job_data['work_class']}
        Industry: {st.session_state.job_data['business_type']}"""

        if feedback:
            content += f"\n\nIncorporate this feedback: {feedback}"

        # Use vLLM API with streaming
        url = "http://localhost:8000/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0.7,
           #"max_tokens": 1000,
            "stream": True
        }

        # Initialize variables
        display_placeholder = st.empty()
        buffer = ""
        complete_text = ""
        thinking_complete = False

        with requests.post(url, json=data, headers=headers, stream=True) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    try:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            json_data = json.loads(line[6:])

                            if 'choices' in json_data and len(json_data['choices']) > 0:
                                chunk = json_data['choices'][0].get('delta', {}).get('content', '')
                                buffer += chunk

                                # Check for thinking parts
                                if '</think>' in buffer and not thinking_complete:
                                    parts = buffer.split('</think>')
                                    if len(parts) > 1:
                                        # Reset buffer to content after thinking
                                        buffer = parts[-1]
                                        thinking_complete = True

                                if thinking_complete:
                                    # Clean the accumulated text
                                    cleaned_text = clean_description(buffer)

                                    if '[city]' in cleaned_text:
                                        cleaned_text = cleaned_text.replace('[city]', city)
                                    if '[City]' in cleaned_text:
                                        cleaned_text = cleaned_text.replace('[City]', city)
                                    if 'city' in cleaned_text:
                                        cleaned_text = cleaned_text.replace('city', city)
                                    if '[country]' in cleaned_text:
                                        cleaned_text = cleaned_text.replace('[country]', country)
                                    if '[Country]' in cleaned_text:
                                        cleaned_text = cleaned_text.replace('[Country]', country)

                                    # Format for display in Quill
                                    formatted_text = cleaned_text

                                    # Format for display
                                    formatted_text = cleaned_text

                                    # Only update display if we have a complete section or punctuation
                                    if (chunk.endswith('\n') or
                                            chunk.endswith('.') or
                                            chunk.endswith(':') or
                                            chunk.endswith('!') or
                                            chunk.endswith('?')):
                                        # For preview in the UI
                                        display_placeholder.markdown(formatted_text)
                                        complete_text = formatted_text

                    except json.JSONDecodeError:
                        continue

        # Format the final text properly for the rich text editor

        # Final update with properly formatted text for display
        display_placeholder.markdown(complete_text)

        end_time = time.time()
        st.session_state.generation_time = end_time - start_time

        # Return the HTML-formatted text for the quill editor
        return complete_text

    except Exception as e:
        st.error(f"Error generating job description: {str(e)}")
        return None


def clean_description(text):
    """Clean and format the description text."""
    # Remove any XML tags and styling
    text = re.sub(r'<[^>]+>', '', text)

    # Format bullet points consistently
    text = text.replace('- ', '• ')
    text = text.replace('* ', '• ')
    text = text.replace('•  ', '• ')
    text = text.replace('*•', ' ')


    # Clean up markdown and formatting
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'###\s*', '', text)
    text = re.sub(r'#\s*', '', text)

    # Clean up spacing
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)

    # Ensure proper spacing after punctuation
    text = re.sub(r'([.!?:])(\w)', r'\1 \2', text)

    return text.strip()


# Update the generate_arabic_job_description function to ensure RTL formatting
def clean_arabic_description(text):
    """Clean and format Arabic description text with RTL support."""
    # Remove any XML tags and styling
    text = re.sub(r'<[^>]+>', '', text)

    # Add RTL mark to ensure proper direction
    text = '\u200F' + text  # Add RTL mark at the beginning

    # Format bullet points consistently
    text = text.replace('- ', '• ')
    text = text.replace('* ', '• ')
    text = text.replace('•  ', '• ')

    # Clean up spacing around Arabic punctuation
    text = re.sub(r'\s*([،؛؟!])\s*', r'\1 ', text)

    # Clean up multiple spaces and newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)

    # Ensure proper RTL formatting for mixed content
    text = re.sub(r'([A-Za-z])\s*([؟،؛])', r'\1\2', text)

    return text.strip()

def generate_arabic_job_description(feedback=None):
    try:
        start_time = time.time()

        # OpenRouter API configuration
        API_KEY = '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'
        API_URL = 'https://openrouter.ai/api/v1/chat/completions'

        # Define the headers
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }

        system_prompt = """You are an experienced HR professional crafting detailed job descriptions in Arabic write it from right to left .
Create a clear and professional job description following this EXACT format in Arabic.
Important: Generate ONLY the job description text in Arabic, no other text or analysis.

المسمى الوظيفي: [title]

الموقع: [city], [country]

نوع العمل: [type]

المستوى: [level]

القطاع: [industry]

نظرة عامة:
[2-3 sentences about the role and its impact]

المسؤوليات الرئيسية:
• [Clear, action-oriented responsibility]
• [Clear, action-oriented responsibility]
• [Clear, action-oriented responsibility]
• [Clear, action-oriented responsibility]
• [Clear, action-oriented responsibility]

المؤهلات المطلوبة:
• [Specific required qualification]
• [Specific required qualification]
• [Specific required qualification]
• [Specific required qualification]

المهارات والخبرات:
• [Specific skill or experience]
• [Specific skill or experience]
• [Specific skill or experience]
• [Specific skill or experience]

متطلبات إضافية:
• [Additional requirement]
• [Additional requirement]
• [Additional requirement]

بيئة العمل:
[Brief description of work environment and culture]"""

        # Format content
        city = st.session_state.job_data['city']
        country = st.session_state.job_data['country']
        content = f"""Create a professional job description in Arabic for:
        Position: {st.session_state.job_data['position']}
        Location: {city}, {country}
        Work Type: {st.session_state.job_data['work_type']}
        Level: {st.session_state.job_data['work_class']}
        Industry: {st.session_state.job_data['business_type']}"""

        if feedback:
            content += f"\n\nIncorporate this feedback: {feedback}"

        # Prepare the request data
        data = {
            "model": "google/gemini-2.0-flash-lite-preview-02-05:free",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0.7,
            "stream": True
        }

        # Initialize variables
        display_placeholder = st.empty()
        buffer = ""
        complete_text = ""

        # Make the streaming request
        with requests.post(API_URL, json=data, headers=headers, stream=True) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    try:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            json_data = json.loads(line[6:])

                            if 'choices' in json_data and len(json_data['choices']) > 0:
                                chunk = json_data['choices'][0].get('delta', {}).get('content', '')
                                buffer += chunk

                                # Clean the accumulated text
                                cleaned_text = clean_arabic_description(buffer)

                                # Replace city placeholder
                                if '[city]' in cleaned_text:
                                    cleaned_text = cleaned_text.replace('[city]', city)
                                if '[City]' in cleaned_text:
                                    cleaned_text = cleaned_text.replace('[City]', city)
                                if '[country]' in cleaned_text:
                                    cleaned_text = cleaned_text.replace('[country]', country)
                                if '[Country]' in cleaned_text:
                                    cleaned_text = cleaned_text.replace('[Country]', country)

                                # Format for display
                                formatted_text = cleaned_text

                                # Update display on complete sections or punctuation
                                if (chunk.endswith('\n') or
                                        chunk.endswith('.') or
                                        chunk.endswith(':') or
                                        chunk.endswith('!') or
                                        chunk.endswith('؟') or
                                        chunk.endswith('.')):
                                    display_placeholder.markdown(formatted_text)
                                    complete_text = formatted_text

                    except json.JSONDecodeError:
                        continue

        # Final update with properly formatted text
        display_placeholder.markdown(complete_text)

        end_time = time.time()
        st.session_state.generation_time = end_time - start_time

        return complete_text

    except Exception as e:
        st.error(f"Error generating Arabic job description: {str(e)}")
        return None


def handle_generate_click():
    if not st.session_state.job_data['position']:
        st.error("Please fill in the position first.")
        return

    with st.spinner("Generating job description..."):
        if st.session_state.job_data['language'] == 'Arabic':
            formatted_description = generate_arabic_job_description()
            if formatted_description:
                st.session_state.arabic_description = formatted_description
                st.session_state.is_generated = True
                st.success("Arabic job description generated successfully!")
            else:
                st.error("Failed to generate the Arabic job description.")
        else:
            formatted_description = generate_job_description()
            if formatted_description:
                st.session_state.english_description = formatted_description
                st.session_state.is_generated = True
                st.success("Job description generated successfully!")
            else:
                st.error("Failed to generate the job description.")

# Update the handle_feedback_submission function
def handle_feedback_submission():
    if not st.session_state.job_data['feedback']:
        st.warning("Please provide feedback before updating.")
        return

    with st.spinner("Updating job description with feedback..."):
        if st.session_state.job_data['language'] == 'Arabic':
            updated_description = generate_arabic_job_description(
                feedback=st.session_state.job_data['feedback']
            )
            if updated_description:
                # Store the complete description in feedback
                st.session_state.arabic_feedback_description = updated_description
                st.session_state.job_data['feedback'] = ""
                st.session_state.is_generated = True
                st.success("Arabic job description updated successfully!")
            else:
                st.error("Failed to update the Arabic job description.")
        else:
            updated_description = generate_job_description(
                feedback=st.session_state.job_data['feedback']
            )
            if updated_description:
                st.session_state.english_feedback_description = updated_description
                st.session_state.job_data['feedback'] = ""
                st.session_state.is_generated = True
                st.success("Job description updated successfully!")
            else:
                st.error("Failed to update the job description.")



def main():
    initialize_session_state()

    st.title("New Job Post")

    tabs = st.tabs([
        "1️⃣ Basic Information",
        "2️⃣ Job Description"
    ])

    # Basic Information Tab
    # Basic Information Tab
    with tabs[0]:
        st.header("Basic Information")

        # Add language selection at the top
        st.subheader("Language")
        language = st.radio(
            "Select Job Description Language",
            options=["English", "Arabic"],
            horizontal=True,
            key="language_input",
            index=0 if st.session_state.job_data['language'] == 'English' else 1
        )
        st.session_state.job_data['language'] = language

        # Position
        st.subheader("Position")
        position = st.text_input(
            "Position Name",
            value=st.session_state.job_data['position'],
            key="position_input"
        )
        st.session_state.job_data['position'] = position

        # Display Name
        st.subheader("Display Name")
        display_name = st.text_input(
            "Display Name",
            value=st.session_state.job_data['display_name'],
            key="display_name_input"
        )
        st.session_state.job_data['display_name'] = display_name

        # Recruitment Type
        st.subheader("Recruitment Type")
        recruitment_type = st.radio(
            "Select Recruitment Type",
            options=["External", "Internal", "Both"],
            horizontal=True,
            key="recruitment_type_input"
        )
        st.session_state.job_data['recruitment_type'] = recruitment_type

        # Number of Vacancies
        st.subheader("Number of Vacancies")
        vacancies = st.number_input(
            "Number of Positions",
            min_value=1,
            value=st.session_state.job_data['vacancies'],
            key="vacancies_input"
        )
        st.session_state.job_data['vacancies'] = vacancies

        # Dates
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Recruitment Start Date")
            start_date = st.date_input(
                "Select Start Date",
                value=st.session_state.job_data['start_date'],
                key="start_date_input"
            )
            st.session_state.job_data['start_date'] = start_date

        with col2:
            st.subheader("Target Hiring Date")
            target_date = st.date_input(
                "Select Target Date",
                value=st.session_state.job_data['target_date'],
                key="target_date_input"
            )
            st.session_state.job_data['target_date'] = target_date

        # Job Post Details
        st.header("Job Post Details")
        st.caption("These details will be viewed for external applicants when the vacancy is shared in job portals")

        # Country and City
        col1, col2 = st.columns(2)
        with col1:
            country = st.text_input(
                "Country",
                value=st.session_state.job_data['country'],
                key="country_input"
            )
            st.session_state.job_data['country'] = country

        with col2:
            city = st.text_input(
                "City",
                value=st.session_state.job_data['city'],
                key="city_input"
            )
            st.session_state.job_data['city'] = city

        # Work Type and Class
        col1, col2 = st.columns(2)
        with col1:
            work_type = st.selectbox(
                "Work Type",
                options=["Full Time", "Part Time", "Contract", "Temporary"],
                key="work_type_input"
            )
            st.session_state.job_data['work_type'] = work_type

        with col2:
            work_class = st.selectbox(
                "Work Class",
                options=["Entry Level", "Mid Level", "Senior Level", "Top Management"],
                key="work_class_input"
            )
            st.session_state.job_data['work_class'] = work_class

        # Business Type
        business_type = st.selectbox(
            "Business Type",
            options=["Accounting", "IT", "Sales", "Marketing", "Human Resources", "Operations"],
            key="business_type_input"
        )
        st.session_state.job_data['business_type'] = business_type

        # Save button
        if st.button("Save Basic Information", type="primary"):
            st.success("Basic Information saved successfully!")

        # Job Description Tab
    with tabs[1]:
        st.header("Job Description")

        if not st.session_state.job_data['position']:
            st.warning("Please fill in the basic information first.")
        else:
            # Generate button
            if st.button("Generate Job Description", type="primary", use_container_width=True):
                handle_generate_click()

            # Conditionally show either English or Arabic description based on selected language
            if st.session_state.job_data['language'] == 'English':
                st.subheader("English Version")
                english_description = st_quill(
                    value=st.session_state.english_description,
                    placeholder="Click 'Generate Job Description' above to create a description...",
                    html=True,
                    key="english_quill",
                    toolbar=[
                        ['bold', 'italic', 'underline'],
                        [{'list': 'ordered'}, {'list': 'bullet'}],
                        [{'header': [1, 2, 3, False]}],
                        [{'direction': 'rtl'}, {'direction': 'ltr'}],
                        [{'align': ['right', 'center', 'left']}],
                        ['clean']
                    ]
                )

                # Store any manual edits back to session state
                if english_description != st.session_state.english_description:
                    st.session_state.english_description = english_description

            else:  # Arabic selected
                st.subheader("النسخة العربية")  # Arabic Version header
                arabic_description = st_quill(
                    value=st.session_state.arabic_description,
                    placeholder="انقر على 'Generate Job Description' أعلاه لإنشاء الوصف الوظيفي...",
                    html=True,
                    key="arabic_quill",
                    toolbar=[
                        ['bold', 'italic', 'underline'],
                        [{'list': 'ordered'}, {'list': 'bullet'}],
                        [{'header': [1, 2, 3, False]}],
                        [{'direction': 'rtl'}, {'direction': 'ltr'}],
                        [{'align': ['right', 'center', 'left']}],
                        ['clean']
                    ],

                )

                # Store any manual edits to Arabic description
                if arabic_description != st.session_state.arabic_description:
                    st.session_state.arabic_description = arabic_description

        if st.session_state.is_generated:
            # Feedback input
            feedback = st.text_area(
                "Enter your feedback to improve the job description" if st.session_state.job_data[
                                                                            'language'] == 'English'
                else "أدخل ملاحظاتك لتحسين الوصف الوظيفي",
                value="",
                key="feedback_input",
                placeholder="Provide specific feedback..." if st.session_state.job_data['language'] == 'English'
                else "قدم ملاحظات محددة..."
            )
            st.session_state.job_data['feedback'] = feedback

            if st.button("Update with Feedback", type="secondary"):
                handle_feedback_submission()

            # Show feedback descriptions if they exist
            if st.session_state.english_feedback_description or st.session_state.arabic_feedback_description:
                st.markdown("---")
                st.subheader("Feedback Results")

                if st.session_state.job_data['language'] == 'English':
                    st.subheader("Updated Description with Feedback")

                    english_feedback = st_quill(
                        value=st.session_state.english_feedback_description,
                        placeholder="Updated description will appear here after providing feedback...",
                        html=True,
                        key="english_feedback_quill",
                        toolbar=[
                            ['bold', 'italic', 'underline'],
                            [{'list': 'ordered'}, {'list': 'bullet'}],
                            [{'header': [1, 2, 3, False]}],
                            [{'direction': 'rtl'}, {'direction': 'ltr'}],
                            [{'align': ['right', 'center', 'left']}],
                            ['clean']
                        ]

                    )

                    # Store any manual edits back to session state
                    if english_feedback != st.session_state.english_feedback_description:
                        st.session_state.english_feedback_description = english_feedback
                else:
                    st.subheader("الوصف المحدث بناءً على الملاحظات")  # Updated Description with Feedback in Arabic
                    arabic_feedback = st_quill(
                        value=st.session_state.arabic_feedback_description,
                        placeholder="سيظهر الوصف المحدث هنا بعد تقديم الملاحظات...",
                        html=True,
                        key="arabic_feedback_quill",
                        toolbar=[
                            ['bold', 'italic', 'underline'],
                            [{'list': 'ordered'}, {'list': 'bullet'}],
                            [{'header': [1, 2, 3, False]}],
                            [{'direction': 'rtl'}, {'direction': 'ltr'}],
                            [{'align': ['right', 'center', 'left']}],
                            ['clean']

                        ]
                    )

if __name__ == "__main__":
    main()
# we need to run this code on the terminal
# python -m vllm.entrypoints.openai.api_server     --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B     --host 0.0.0.0     --port 8000     --gpu-memory-utilization 0.95     --max-model-len 8192     --trust-remote-code
