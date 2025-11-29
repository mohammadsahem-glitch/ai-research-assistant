# Deployment Guide - AI Research Assistant

This guide will help you deploy your AI Research Assistant to the internet using various platforms.

## Prerequisites

- Git installed on your system
- GitHub account (for code hosting)
- Your OpenAI API key
- Your SerpAPI key

## Option 1: Deploy to Railway (Recommended - Easiest)

Railway is the easiest platform for deployment with a generous free tier.

### Steps:

1. **Create a GitHub repository**
   ```bash
   cd my_ai_agent
   git init
   git add .
   git commit -m "Initial commit - AI Research Assistant"
   ```

2. **Push to GitHub**
   - Create a new repository on GitHub
   - Follow GitHub's instructions to push your code:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/ai-research-assistant.git
   git branch -M main
   git push -u origin main
   ```

3. **Deploy on Railway**
   - Go to [railway.app](https://railway.app)
   - Click "Start a New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway will auto-detect Flask and deploy

4. **Add Environment Variables**
   - In Railway dashboard, go to your project
   - Click on "Variables" tab
   - Add:
     - `OPENAI_API_KEY`: your_openai_key
     - `SERPAPI_API_KEY`: your_serpapi_key
   - Railway will automatically redeploy

5. **Access Your App**
   - Railway will provide a URL like: `https://your-app.railway.app`
   - First user to register will need to create an account

### Railway Features:
- ✅ Free tier: 500 hours/month
- ✅ Auto-deploy on git push
- ✅ Free SSL certificate
- ✅ Easy database management

---

## Option 2: Deploy to Render

Render offers a free tier with automatic deploys from GitHub.

### Steps:

1. **Push code to GitHub** (same as Railway step 1-2)

2. **Deploy on Render**
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: ai-research-assistant
     - **Environment**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn app:app`

3. **Add Environment Variables**
   - In Render dashboard, go to "Environment"
   - Add:
     - `OPENAI_API_KEY`
     - `SERPAPI_API_KEY`

4. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes)

### Render Features:
- ✅ Free tier available
- ✅ Auto-deploy on git push
- ✅ Free SSL certificate
- ⚠️ Free tier spins down after inactivity (slow first load)

---

## Option 3: Deploy to Heroku

Heroku is a well-established platform but requires a credit card for verification.

### Steps:

1. **Install Heroku CLI**
   ```bash
   # Download from: https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login and Create App**
   ```bash
   heroku login
   cd my_ai_agent
   heroku create your-ai-assistant
   ```

3. **Set Environment Variables**
   ```bash
   heroku config:set OPENAI_API_KEY=your_key
   heroku config:set SERPAPI_API_KEY=your_key
   ```

4. **Deploy**
   ```bash
   git push heroku main
   ```

5. **Open Your App**
   ```bash
   heroku open
   ```

### Heroku Features:
- ✅ Reliable and mature platform
- ✅ Easy to use CLI
- ✅ Free SSL certificate
- ⚠️ Requires credit card verification
- ⚠️ Free tier limited to 550-1000 dyno hours/month

---

## Option 4: PythonAnywhere

Good for simple deployments, but may require manual configuration.

### Steps:

1. **Create Account**
   - Go to [pythonanywhere.com](https://www.pythonanywhere.com)
   - Sign up for free account

2. **Upload Files**
   - Use "Files" tab to upload your project
   - Or use bash console to git clone

3. **Install Dependencies**
   - Open a Bash console
   ```bash
   mkvirtualenv myenv --python=/usr/bin/python3.10
   cd my_ai_agent
   pip install -r requirements.txt
   ```

4. **Configure Web App**
   - Go to "Web" tab
   - Click "Add a new web app"
   - Choose "Manual configuration" → Python 3.10
   - Set:
     - **Source code**: `/home/yourusername/my_ai_agent`
     - **Working directory**: `/home/yourusername/my_ai_agent`
     - **WSGI file**: Edit to point to your app

5. **Set Environment Variables**
   - In web app settings, add environment variables in .env file

6. **Reload App**

### PythonAnywhere Features:
- ✅ Free tier available
- ✅ Good for learning
- ⚠️ More manual setup required
- ⚠️ Free tier has CPU limitations

---

## Important Notes for All Platforms

### 1. Database
The app uses SQLite which works on all platforms but has limitations:
- For production with multiple users, consider PostgreSQL
- Most platforms offer free PostgreSQL databases
- To switch to PostgreSQL:
  ```python
  # In app.py, change database URI:
  app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
  ```

### 2. File Uploads
- Most platforms have ephemeral file systems
- Uploaded files may be deleted on restart
- For production, consider cloud storage (AWS S3, Cloudinary)

### 3. Security
- Use strong SECRET_KEY in production:
  ```python
  # In app.py:
  app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24))
  ```
- Set SECRET_KEY environment variable on your platform

### 4. API Keys
- NEVER commit API keys to git
- Always use environment variables
- Add .env to .gitignore (already done)

### 5. HTTPS
- All recommended platforms provide free SSL/HTTPS
- Your app will be accessible via https://

---

## Testing Before Deployment

Before deploying, test locally with gunicorn:

```bash
pip install gunicorn
gunicorn app:app
```

Visit http://127.0.0.1:8000

---

## Monitoring and Logs

- **Railway**: View logs in dashboard
- **Render**: "Logs" tab in service dashboard
- **Heroku**: `heroku logs --tail`
- **PythonAnywhere**: "Log files" section

---

## Recommended Deployment Path

**For Beginners**: Railway (easiest, generous free tier)
**For Production**: Render or Heroku (more features, better uptime)
**For Learning**: PythonAnywhere (great tutorials)

---

## After Deployment

1. Visit your deployed URL
2. Register first user account
3. Test all features:
   - Login/logout
   - Create chat sessions
   - Upload documents
   - Ask questions
   - Test different agent modes
   - Verify news widgets work
   - Test table rendering

4. Share your URL with users!

---

## Troubleshooting

### "Application Error" on load
- Check logs for errors
- Verify environment variables are set
- Ensure all dependencies in requirements.txt

### Database errors
- Platform may reset database on deployment
- For persistence, use managed database service

### Slow first load
- Free tiers often "sleep" after inactivity
- First request wakes the app (10-30 seconds)
- Subsequent requests are fast

### File upload failures
- Check file size limits (currently 50MB)
- Verify uploads/ directory is created
- Consider cloud storage for production

---

## Support

For issues:
1. Check platform documentation
2. Review application logs
3. Verify environment variables
4. Test locally first

---

## Next Steps

- Set up custom domain (most platforms support this)
- Add email notifications
- Implement password reset
- Add user profiles
- Set up monitoring/analytics
- Configure backup strategy

Good luck with your deployment! 🚀
