interface UrlCheckResponse {
  blocked: boolean;
  probability: number;
  risk_level: string;
  url: string;
}

export const checkUrl = async (url: string): Promise<UrlCheckResponse> => {
  try {
    const response = await fetch('http://localhost:8000/api/check-url', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: `url=${encodeURIComponent(url)}`,
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    return await response.json();
  } catch (error) {
    console.error('Error checking URL:', error);
    throw error;
  }
};

export const retrainModel = async (): Promise<{ status: string }> => {
  try {
    const response = await fetch('http://localhost:8000/api/retrain', {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    return await response.json();
  } catch (error) {
    console.error('Error retraining model:', error);
    throw error;
  }
};
