# job-description

# Job Posting System

A streamlined job posting system built with Streamlit that generates professional job descriptions in both English and Arabic using advanced language models. The system provides an intuitive interface for creating, editing, and managing job postings with real-time AI-powered description generation.

## Features

- Bilingual support (English and Arabic)
- Real-time job description generation
- Interactive rich text editor
- Feedback-based description refinement
- Comprehensive job posting management
-  support for Arabic content
- Streamlined user interface

## Models Used

The system uses two different language models for generating job descriptions:

1. **DeepSeek-R1-Distill-Qwen-7B** (English Descriptions)
   - A distilled version of the Qwen model optimized for efficiency
   - Deployed using vLLM for improved performance
   - Used for generating English job descriptions

2. **Google Gemini 2.0 Flash Lite** (Arabic Descriptions)
   - Accessed through OpenRouter API
   - Optimized for Arabic language generation
   - Used for generating Arabic job descriptions

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for optimal performance)
- At least 16GB of GPU memory
- Internet connection for API access

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd job-posting-system
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Environment Setup

1. Set up your OpenRouter API key:
   - Sign up at openrouter.ai
   - Get your API key
   - Set it in the code or as an environment variable:
```bash
export OPENROUTER_API_KEY='your-api-key'
```

2. Deploy the vLLM server for DeepSeek model:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --trust-remote-code
```

## Running the Application

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Access the application in your web browser at `http://localhost:8501`

## Usage

1. **Basic Information Tab**
   - Select language (English/Arabic)
   - Fill in job details (position, location, type, etc.)
   - Save basic information

2. **Job Description Tab**
   - Click "Generate Job Description" to create initial description
   - Edit description using the rich text editor
   - Provide feedback for refinements
   - Update description based on feedback

## Important Notes

1. **GPU Requirements**
   - The DeepSeek model requires significant GPU memory
   - Recommended: GPU with 16GB+ memory
   - CPU fallback available but significantly slower

2. **API Usage**
   - Arabic generation uses OpenRouter API
   - Ensure API key is valid and has sufficient credits
   - Monitor API usage to avoid interruptions

3. **Performance Optimization**
   - vLLM server should be running before starting the application
   - Adjust GPU memory utilization as needed
   - Consider batch size settings for your hardware

## Error Handling

The system includes comprehensive error handling for:
- API failures
- Model generation issues
- File processing errors
- Invalid input validation

## Best Practices

1. **Job Description Generation**
   - Provide detailed basic information for better results
   - Use specific feedback for refinements
   - Review and edit generated content as needed

2. **System Resources**
   - Monitor GPU memory usage
   - Restart vLLM server if performance degrades
   - Clear browser cache regularly for optimal performance

3. **Content Management**
   - Save important descriptions regularly
   - Use feedback system for incremental improvements
   - Validate generated content before finalizing

## Troubleshooting

Common issues and solutions:

1. **vLLM Server Issues**
   - Ensure CUDA is properly installed
   - Check GPU memory availability
   - Verify port 8000 is available

2. **API Connection Problems**
   - Verify internet connectivity
   - Check API key validity
   - Monitor API rate limits

3. **Generation Failures**
   - Ensure all required fields are filled
   - Check for valid input formatting
   - Verify model server status


## to do
A Flask-based API with webhook support is required for production deployment, allowing real-time processing and integration with external systems.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request
4. Follow coding standards
5. Include tests where applicable
