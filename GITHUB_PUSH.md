# 🚀 How to Push to GitHub

I have already initialized the local Git repository and committed all your files.

To push this to GitHub, follow these steps:

## 1. Create a New Repository on GitHub
1.  Go to [github.com/new](https://github.com/new).
2.  Name it `hyperliquid-trading-bot` (or whatever you prefer).
3.  **Do NOT** initialize with README, .gitignore, or License (we already have them).
4.  Click **Create repository**.

## 2. Push Your Code
Copy the commands shown on GitHub under "…or push an existing repository from the command line", or use these (replace `YOUR_USERNAME`):

```bash
# Add the remote repository
git remote add origin https://github.com/YOUR_USERNAME/hyperliquid-trading-bot.git

# Rename branch to main (if not already)
git branch -M main

# Push your code
git push -u origin main
```

## 3. Verify
Refresh your GitHub page. You should see all your files!

> [!IMPORTANT]
> Your `.env` file (containing keys) and `data/` directory are **ignored** by `.gitignore` and will NOT be pushed. This is for your security.
