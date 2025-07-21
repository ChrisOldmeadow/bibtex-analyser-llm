#!/bin/bash

# Heroku Deployment Script for Bibtex Analyzer Dashboard
# This script helps deploy the dashboard to Heroku

set -e

APP_NAME="bibtex-analyzer-dashboard"
HEROKU_REGION="us"

echo "🚀 Deploying Bibtex Analyzer Dashboard to Heroku"
echo "=================================================="

# Check if Heroku CLI is installed
if ! command -v heroku &> /dev/null; then
    echo "❌ Heroku CLI is not installed. Please install it first:"
    echo "   Visit: https://devcenter.heroku.com/articles/heroku-cli"
    exit 1
fi

# Check if user is logged in to Heroku
if ! heroku auth:whoami &> /dev/null; then
    echo "❌ Not logged in to Heroku. Please login first:"
    echo "   Run: heroku login"
    exit 1
fi

echo "✅ Heroku CLI is installed and you are logged in"

# Create Heroku app if it doesn't exist
if ! heroku apps:info $APP_NAME &> /dev/null; then
    echo "📱 Creating new Heroku app: $APP_NAME"
    heroku create $APP_NAME --region=$HEROKU_REGION
else
    echo "📱 Using existing Heroku app: $APP_NAME"
fi

echo "🔧 Setting up environment variables..."

# Set required environment variables
echo "⚠️  IMPORTANT: You need to set your OpenAI API key"
echo "   Run: heroku config:set OPENAI_API_KEY=your_api_key_here -a $APP_NAME"
echo ""

# Set optional environment variables with defaults
heroku config:set \
    PYTHONPATH="/app" \
    DEFAULT_MODEL="gpt-3.5-turbo" \
    LOG_LEVEL="INFO" \
    -a $APP_NAME

echo "📦 Adding Heroku remote (if not already added)..."
if ! git remote get-url heroku &> /dev/null; then
    heroku git:remote -a $APP_NAME
else
    echo "   Heroku remote already exists"
fi

echo "🚢 Deploying to Heroku..."
git add .
git commit -m "Prepare for Heroku deployment" || echo "No changes to commit"
git push heroku main

echo "🌐 Opening the deployed app..."
heroku open -a $APP_NAME

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Set your OpenAI API key: heroku config:set OPENAI_API_KEY=your_key -a $APP_NAME"
echo "   2. View logs: heroku logs --tail -a $APP_NAME"
echo "   3. Scale dynos if needed: heroku ps:scale web=1 -a $APP_NAME"
echo ""
echo "🔗 App URL: https://$APP_NAME.herokuapp.com"
echo "🔧 Heroku Dashboard: https://dashboard.heroku.com/apps/$APP_NAME"