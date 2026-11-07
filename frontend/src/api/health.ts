import axios from 'axios'

export interface HealthResponse {
  status: string
  application: string
}

const apiClient = axios.create({ baseURL: '/api', timeout: 5000 })

export async function getHealthStatus(): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>('/health')
  return response.data
}
