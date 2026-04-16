// Vercel serverless function — polls GitHub Actions workflow status
// Returns the latest workflow run status so the frontend knows when refresh is done.

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ ok: false, error: 'Method not allowed' });
  }

  const token = process.env.GITHUB_TOKEN;
  const repo = process.env.GITHUB_REPO;

  if (!token || !repo) {
    return res.status(500).json({
      ok: false,
      error: 'Server not configured.',
    });
  }

  try {
    // Get the most recent workflow run for refresh.yml
    const response = await fetch(
      `https://api.github.com/repos/${repo}/actions/workflows/refresh.yml/runs?per_page=1`,
      {
        headers: {
          Authorization: `Bearer ${token}`,
          Accept: 'application/vnd.github.v3+json',
        },
      }
    );

    if (!response.ok) {
      const text = await response.text();
      return res.status(response.status).json({
        ok: false,
        error: `GitHub API returned ${response.status}: ${text}`,
      });
    }

    const data = await response.json();
    const run = data.workflow_runs && data.workflow_runs[0];

    if (!run) {
      return res.status(200).json({ ok: true, status: 'none', message: 'No workflow runs found' });
    }

    return res.status(200).json({
      ok: true,
      status: run.status,           // queued, in_progress, completed
      conclusion: run.conclusion,   // success, failure, null (if still running)
      started_at: run.run_started_at,
      updated_at: run.updated_at,
      html_url: run.html_url,
    });
  } catch (err) {
    return res.status(500).json({ ok: false, error: err.message });
  }
}
