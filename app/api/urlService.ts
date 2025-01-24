const API_BASE_URL = "http://localhost:8000/api";

export interface Activity {
  url: string
  timestamp: string
  action: 'blocked' | 'allowed' | 'visited' | 'checking' | 'error'
  category?: string
  risk_level?: string
}

export interface DashboardStats {
  total_sites: number
  blocked_sites: number
  allowed_sites: number
  visited_sites: number
  recent_activities: Activity[]
  daily_stats: { [key: string]: number }
}

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.text();
    throw new ApiError(response.status, error);
  }
  return response.json();
}

export const urlService = {
  async checkUrl(url: string): Promise<{
    blocked: boolean
    probability: number
    risk_level: string
    category: string
    url: string
  }> {
    const formData = new FormData();
    formData.append('url', url);

    const response = await fetch(`${API_BASE_URL}/check-url`, {
      method: 'POST',
      body: formData
    });

    return handleResponse(response);
  },

  async getDashboardStats(): Promise<DashboardStats> {
    const response = await fetch(`${API_BASE_URL}/dashboard/stats`, {
      headers: {
        'Accept': 'application/json'
      }
    });

    return handleResponse(response);
  },

  async recordActivity(activity: {
    url: string
    action: string
    category?: string
    timestamp?: string
  }): Promise<Activity> {
    const response = await fetch(`${API_BASE_URL}/activity`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(activity),
    });

    return handleResponse(response);
  },

  async retrainModel(): Promise<{ status: string }> {
    const response = await fetch(`${API_BASE_URL}/retrain`, {
      method: 'POST',
      headers: {
        'Accept': 'application/json'
      }
    });

    return handleResponse(response);
  },

  async fetchYouTubeActivity(): Promise<Activity[]> {
    const response = await fetch(`${API_BASE_URL}/youtube-activity`, {
      headers: {
        'Accept': 'application/json'
      }
    });

    return handleResponse(response);
  },

  async retryWithBackoff<T>(
    operation: () => Promise<T>,
    retries = 3,
    delay = 1000,
    backoffRate = 2
  ): Promise<T> {
    let currentDelay = delay;

    for (let i = 0; i < retries; i++) {
      try {
        return await operation();
      } catch (error) {
        if (i === retries - 1) throw error;

        console.error(`Attempt ${i + 1} failed, retrying in ${currentDelay}ms...`);
        await new Promise(resolve => setTimeout(resolve, currentDelay));
        currentDelay *= backoffRate;
      }
    }

    throw new Error('Operation failed after maximum retries');
  }
};

// Export a helper for checking extension connectivity
export async function checkBackendConnectivity(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/dashboard/stats`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' }
    });
    return response.ok;
  } catch {
    return false;
  }
}
