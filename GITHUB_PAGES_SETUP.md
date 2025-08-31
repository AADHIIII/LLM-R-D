# 🌐 GitHub Pages Setup Guide

## 📋 Step-by-Step Instructions

### Step 1: Add Your API Key to GitHub Secrets

1. **Go to your GitHub repository**: https://github.com/AADHIIII/LLM-R-D
2. **Click on "Settings"** (top menu of your repo)
3. **Click on "Secrets and variables"** → **"Actions"** (left sidebar)
4. **Click "New repository secret"**
5. **Add these secrets one by one:**

   **Secret 1:**
   - Name: `GEMINI_API_KEY`
   - Value: `AIzaSyDVye-fWaH0ju-JAeWq2ad3JqFBtmrZNqE`

   **Secret 2:**
   - Name: `SECRET_KEY`
   - Value: `my-super-secret-key-12345`

   **Secret 3:**
   - Name: `JWT_SECRET_KEY`
   - Value: `my-jwt-secret-key-67890`

### Step 2: Enable GitHub Pages

1. **Still in Settings**, scroll down to **"Pages"** (left sidebar)
2. **Under "Source"**, select **"GitHub Actions"**
3. **Click "Save"**

### Step 3: Trigger Deployment

1. **Go to "Actions"** tab in your repo
2. **You should see the "Deploy to GitHub Pages" workflow**
3. **If it hasn't run automatically, click "Run workflow"**

### Step 4: Access Your Live Site

After deployment completes (2-5 minutes):
- **Your site will be live at**: `https://aadhiiii.github.io/LLM-R-D/`

## 🎯 What You'll Get

✅ **Beautiful Web Interface** - Modern, responsive design  
✅ **Gemini AI Integration** - Test prompts with your API key  
✅ **Model Comparison Tools** - Compare different AI models  
✅ **Analytics Dashboard** - Track performance and costs  
✅ **Mobile-Friendly** - Works on all devices  

## 🔧 Troubleshooting

**If deployment fails:**
1. Check the "Actions" tab for error messages
2. Ensure all secrets are added correctly
3. Make sure the repository is public (or you have GitHub Pro)

**If the site loads but API doesn't work:**
1. Check that `GEMINI_API_KEY` secret is set correctly
2. Open browser developer tools to check for errors

## 🚀 Next Steps

Once your site is live, you can:
1. **Test the Gemini integration** with the prompt testing page
2. **Share your live URL** with others
3. **Add more API keys** (OpenAI, Anthropic) for more models
4. **Customize the interface** by modifying the React components

## 📱 Mobile Access

Your platform is fully responsive and works great on:
- 📱 Mobile phones
- 📱 Tablets  
- 💻 Desktops
- 🖥️ Large screens

## 🎉 Congratulations!

You now have a professional LLM optimization platform deployed for free on GitHub Pages! 🚀