# Deploying to GitHub

Since you have the `gh` CLI installed, follow these steps to publish your repository:

### 1. Authenticate with GitHub (if not already done)
```bash
gh auth login
```
Follow the prompts (select GitHub.com, HTTPS or SSH, and login via browser).

### 2. Create the Repository and Push
Run the following command in the project root (`dynamic_portfolio_optimization/`):

```bash
gh repo create dynamic-portfolio-optimization --public --source=. --remote=origin
```
Using `--source=.` will initialize the remote from your current directory.

### 3. Push your Code
```bash
git push -u origin main
```

Your code is now live on GitHub!

---

# Running the Full-Stack App

### Backend (Terminal 1)
```bash
source venv/bin/activate
uvicorn backend.api:app --reload --port 8000
```

### Frontend (Terminal 2)
```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173` to access the AI Portfolio Manager.
