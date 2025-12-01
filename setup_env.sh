#!/bin/bash

echo "====================================="
echo "  LLM Trading Agent - Setup"
echo "====================================="
echo ""
echo "Please enter your OpenAI API key:"
read -r OPENAI_KEY

# Replace the placeholder in .env
sed -i "s/PUT_YOUR_OPENAI_KEY_HERE/$OPENAI_KEY/" .env

echo ""
echo "✓ .env file updated successfully!"
echo ""
echo "To verify (will not show your key):"
echo "  grep -c 'sk-' .env"
echo ""
echo "To test the agent:"
echo "  python agent.py"
echo ""
