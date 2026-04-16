// Vercel serverless function — triggers GitHub Actions workflow
// Env vars needed in Vercel dashboard:
//   GITHUB_TOKEN   — Personal Access Token (classic) with "repo" scope
//   GITHUB_REPO    — e.g. "username/leversens"

export default async function handler(req, res) {
  // Only POST
  if (req.method !== 'POST') {
    return res.status(405).json({ ok: false, error: 'Method not allowed' });
  }

  const token = process.env.GITHUB_TOKEN;
  const repo = process.env.GITHUB_REPO;

  if (!token || !repo) {
    return res.status(500).json({
      ok: false,
      error: 'Server not configured. Set GITHUB_TOKEN and GITHUB_REPO in Vercel env vars.',
    });
  }

  try {
    const response = await fetch(
      `https://api.github.com/repos/${repo}/actions/workflows/refresh.yml/dispatches`,
      {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          Accept: 'application/vnd.github.v3+json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ref: 'main' }),
      }
    );

    if (response.status === 204) {
      return res.status(200).json({ ok: true, message: 'Workflow triggered' });
    } else {
      const text = await response.text();
      return res.status(response.status).json({
        ok: false,
        error: `GitHub API returned ${response.status}: ${text}`,
      });
    }
  } catch (err) {
    return res.status(500).json({ ok: false, error: err.message });
  }
}
