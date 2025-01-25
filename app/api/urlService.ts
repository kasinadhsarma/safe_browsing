const API_BASE_URL = "http://localhost:8000/api";

export interface Activity {
  url: string;
  timestamp: string;
  action: 'blocked' | 'allowed' | 'visited' | 'checking' | 'error' | 'override';
  category: string;
  risk_level: string;
  age_group?: string;
  block_reason?: string;
  ml_scores?: { [key: string]: number };
}

export interface Alert {
  id: string;
  message: string;
  priority: 'high' | 'medium' | 'low';
  timestamp: string;
}

export interface ProtectionStats {
  websites_blocked: number;
  threats_detected: number;
  content_filtered: number;
  protection_score: number;
}

export interface DashboardStats {
  total_sites: number;
  blocked_sites: number;
  allowed_sites: number;
  visited_sites: number;
  recent_activities: Activity[];
  daily_stats: { [key: string]: number };
  protection_stats: ProtectionStats;
  alerts: Alert[];
  ml_model_updates: { id: string; message: string; priority: 'high' | 'medium' | 'low' }[];
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
  const text = await response.text();
  try {
    // Fix malformed JSON by adding missing commas
    const fixedText = text
      .replace(/"([^"]+)"(\d+)/g, '"$1":$2')  // Fix "key"value -> "key":value
      .replace(/}"/g, '},"')  // Fix missing commas between objects
      .replace(/]"/g, '],"'); // Fix missing commas between arrays
    return JSON.parse(fixedText) as T;
  } catch (e) {
    console.error('Error parsing JSON:', e);
    console.error('Original text:', text);
    throw new ApiError(response.status, 'Invalid JSON response from server');
  }
}

export const urlService = {
  async checkUrl(url: string): Promise<{
    blocked: boolean;
    probability: number;
    risk_level: string;
    category: string;
    url: string;
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
    const response = await fetch(`${API_BASE_URL}/stats`, {
      headers: {
        'Accept': 'application/json'
      }
    });

    return handleResponse(response);
  },

  async recordActivity(activity: {
    url: string;
    action: string;
    category: string;
    risk_level: string;
    age_group?: string;
    block_reason?: string;
    ml_scores?: { [key: string]: number };
  }): Promise<Activity> {
    const formData = new FormData();
    formData.append('url', activity.url);
    formData.append('action', activity.action);
    formData.append('category', activity.category);
    formData.append('risk_level', activity.risk_level);
    if (activity.age_group) formData.append('age_group', activity.age_group);
    if (activity.block_reason) formData.append('block_reason', activity.block_reason);
    formData.append('ml_scores', JSON.stringify(activity.ml_scores || {}));

    const response = await fetch(`${API_BASE_URL}/activity`, {
      method: 'POST',
      body: formData
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

  async getRecentActivities(): Promise<Activity[]> {
    const response = await fetch(`${API_BASE_URL}/activities`, {
      headers: {
        'Accept': 'application/json'
      }
    });

    return handleResponse(response);
  },

  async getAlerts(): Promise<Activity[]> {
    const response = await fetch(`${API_BASE_URL}/alerts`, {
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
    const response = await fetch(`${API_BASE_URL}/stats`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' }
    });
    return response.ok;
  } catch {
    return false;
  }
}
