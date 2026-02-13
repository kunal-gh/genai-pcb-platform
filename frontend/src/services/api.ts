import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface DesignRequest {
  prompt: string;
  user_id?: string;
}

export interface DesignResponse {
  design_id: string;
  status: string;
  message: string;
}

export interface DesignStatus {
  design_id: string;
  status: string;
  progress: number;
  current_step: string;
  error_message?: string;
}

export interface DesignDetails {
  id: string;
  prompt: string;
  status: string;
  created_at: string;
  updated_at: string;
  files: DesignFile[];
  verification_results?: VerificationResult[];
}

export interface VerificationResult {
  type: string;
  category: string;
  severity: string;
  message: string;
  suggestion?: string;
  location?: string;
}

export interface DesignFile {
  id: string;
  filename: string;
  file_type: string;
  file_path: string;
}

export const designApi = {
  createDesign: async (request: DesignRequest): Promise<DesignResponse> => {
    const response = await api.post('/api/designs', request);
    return response.data;
  },

  getDesignStatus: async (designId: string): Promise<DesignStatus> => {
    const response = await api.get(`/api/designs/${designId}/status`);
    return response.data;
  },

  getDesignDetails: async (designId: string): Promise<DesignDetails> => {
    const response = await api.get(`/api/designs/${designId}`);
    return response.data;
  },

  listDesigns: async (): Promise<DesignDetails[]> => {
    const response = await api.get('/api/designs');
    return response.data;
  },

  deleteDesign: async (designId: string): Promise<void> => {
    await api.delete(`/api/designs/${designId}`);
  },

  downloadFile: async (designId: string, fileId: string): Promise<Blob> => {
    const response = await api.get(`/api/designs/${designId}/files/${fileId}`, {
      responseType: 'blob',
    });
    return response.data;
  },

  getPreviewImage: async (designId: string, imageType: 'schematic' | 'pcb'): Promise<string | null> => {
    try {
      const response = await api.get(`/api/designs/${designId}/preview/${imageType}`, {
        responseType: 'blob',
      });
      return URL.createObjectURL(response.data);
    } catch (error) {
      return null;
    }
  },

  getVerificationResults: async (designId: string): Promise<VerificationResult[]> => {
    try {
      const response = await api.get(`/api/designs/${designId}/verification`);
      return response.data;
    } catch (error) {
      return [];
    }
  },
};

export default api;
