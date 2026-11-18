import axios from 'axios'

export interface PredictionItem {
  classId: number
  label: string
  confidence: number
}

export interface PredictionResponse extends PredictionItem {
  topPredictions: PredictionItem[]
}

export interface PredictionApiError {
  status: number | null
  message: string
}

function isPredictionItem(value: unknown): value is PredictionItem {
  if (!value || typeof value !== 'object') return false
  const item = value as Record<string, unknown>
  return Number.isInteger(item.classId)
    && typeof item.label === 'string'
    && typeof item.confidence === 'number'
    && Number.isFinite(item.confidence)
}

export function isPredictionResponse(value: unknown): value is PredictionResponse {
  if (!value || typeof value !== 'object') return false
  const response = value as Record<string, unknown>
  return isPredictionItem(response)
    && Array.isArray(response.topPredictions)
    && response.topPredictions.every(isPredictionItem)
}

export async function predictSign(file: File): Promise<PredictionResponse> {
  const formData = new FormData()
  formData.append('file', file)

  try {
    const response = await axios.post<unknown>('/api/predict', formData)
    if (!isPredictionResponse(response.data)) {
      throw {
        status: response.status,
        message: 'Сервер повернув некоректну відповідь розпізнавання.',
      } satisfies PredictionApiError
    }
    return response.data
  } catch (error: unknown) {
    if (axios.isAxiosError(error) && error.response) {
      const status = error.response.status
      const data = error.response.data as { message?: unknown } | undefined
      const serverMessage = typeof data?.message === 'string' ? data.message : null

      if (status === 400) {
        throw { status, message: serverMessage ?? 'Сервер не прийняв зображення.' } satisfies PredictionApiError
      }
      if (status === 413) {
        throw { status, message: 'Розмір зображення перевищує 5 МБ.' } satisfies PredictionApiError
      }
      if (status === 503) {
        throw { status, message: 'ONNX-модель тимчасово недоступна. Спробуйте пізніше.' } satisfies PredictionApiError
      }
      throw { status, message: serverMessage ?? 'Не вдалося виконати розпізнавання.' } satisfies PredictionApiError
    }

    if (axios.isAxiosError(error)) {
      throw {
        status: null,
        message: 'Backend недоступний. Перевірте, чи запущено сервер на localhost:8080.',
      } satisfies PredictionApiError
    }

    if (error && typeof error === 'object' && 'status' in error && 'message' in error) {
      throw error as PredictionApiError
    }

    throw { status: null, message: 'Сталася неочікувана помилка під час розпізнавання.' } satisfies PredictionApiError
  }
}
