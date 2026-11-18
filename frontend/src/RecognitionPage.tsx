import { useEffect, useRef, useState, type DragEvent, type KeyboardEvent } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Container,
  LinearProgress,
  Stack,
  Typography,
} from '@mui/material'
import type { PredictionResponse } from './api/prediction'
import { predictSign } from './api/prediction'
import { formatFileSize, validateImageFile } from './recognitionValidation'

function confidencePercent(confidence: number): string {
  const percent = confidence <= 1 ? confidence * 100 : confidence
  return `${percent.toFixed(2)}%`
}

export default function RecognitionPage() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [previewUrl])

  const selectFile = (selectedFile: File | undefined) => {
    if (loading) return
    const validationError = validateImageFile(selectedFile)
    setResult(null)
    setError(validationError)
    if (validationError || !selectedFile) {
      setFile(null)
      setPreviewUrl((current) => {
        if (current) URL.revokeObjectURL(current)
        return null
      })
      return
    }

    setFile(selectedFile)
    setPreviewUrl((current) => {
      if (current) URL.revokeObjectURL(current)
      return URL.createObjectURL(selectedFile)
    })
  }

  const removeFile = () => {
    if (loading) return
    setFile(null)
    setResult(null)
    setError(null)
    setPreviewUrl((current) => {
      if (current) URL.revokeObjectURL(current)
      return null
    })
    if (inputRef.current) inputRef.current.value = ''
  }

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragActive(false)
    selectFile(event.dataTransfer.files[0])
  }

  const handleDropZoneKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      inputRef.current?.click()
    }
  }

  const recognize = async () => {
    if (!file || loading) return
    const validationError = validateImageFile(file)
    if (validationError) {
      setError(validationError)
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const prediction = await predictSign(file)
      setResult({
        ...prediction,
        topPredictions: [...prediction.topPredictions].sort((a, b) => b.confidence - a.confidence),
      })
    } catch (caughtError: unknown) {
      if (caughtError && typeof caughtError === 'object' && 'message' in caughtError) {
        setError(String(caughtError.message))
      } else {
        setError('Сталася неочікувана помилка під час розпізнавання.')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <Container maxWidth="lg" sx={{ py: { xs: 5, md: 9 } }}>
      <Stack spacing={1} sx={{ mb: 4 }}>
        <Typography variant="h2" sx={{ fontSize: { xs: '2.25rem', md: '3.25rem' } }}>
          Розпізнавання дорожніх знаків
        </Typography>
        <Typography color="text.secondary" sx={{ maxWidth: 680, lineHeight: 1.7 }}>
          Завантажте чітку фотографію одного дорожнього знака формату JPEG або PNG розміром до 5 МБ.
        </Typography>
      </Stack>

      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 1.08fr) minmax(320px, 0.92fr)' }, gap: 3, alignItems: 'start' }}>
        <Card variant="outlined" sx={{ boxShadow: '0 12px 36px rgba(31, 41, 55, 0.06)' }}>
          <CardContent sx={{ p: { xs: 2, sm: 3 } }}>
            <Box
              role="button"
              tabIndex={loading ? -1 : 0}
              aria-label="Завантажити фотографію дорожнього знака"
              onKeyDown={handleDropZoneKeyDown}
              onDragEnter={(event) => { event.preventDefault(); if (!loading) setDragActive(true) }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={(event) => { event.preventDefault(); setDragActive(false) }}
              onDrop={handleDrop}
              onClick={() => { if (!loading) inputRef.current?.click() }}
              sx={{
                minHeight: previewUrl ? 320 : 270,
                border: '1.5px dashed',
                borderColor: dragActive ? 'primary.main' : 'divider',
                borderRadius: 3,
                backgroundColor: dragActive ? 'primary.50' : '#fafbfc',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                overflow: 'hidden',
                cursor: loading ? 'default' : 'pointer',
                transition: 'border-color 180ms ease, background-color 180ms ease',
                '&:focus-visible': { outline: '3px solid', outlineColor: 'primary.light', outlineOffset: 3 },
              }}
            >
              <input
                ref={inputRef}
                hidden
                type="file"
                accept="image/jpeg,image/png"
                onChange={(event) => selectFile(event.target.files?.[0])}
              />
              {previewUrl && file ? (
                <Box component="img" src={previewUrl} alt={`Попередній перегляд: ${file.name}`} sx={{ width: '100%', height: 420, maxHeight: '45vh', objectFit: 'contain', p: 2 }} />
              ) : (
                <Stack spacing={1.5} sx={{ alignItems: 'center', px: 3, textAlign: 'center' }}>
                  <Box sx={{ width: 64, height: 64, borderRadius: '50%', display: 'grid', placeItems: 'center', backgroundColor: 'primary.50', color: 'primary.main', fontSize: 28, fontWeight: 700 }}>
                    +
                  </Box>
                  <Typography variant="h6">Перетягніть фото сюди</Typography>
                  <Typography color="text.secondary">або натисніть, щоб обрати з пристрою</Typography>
                  <Button
                    variant="contained"
                    onClick={(event) => {
                      event.stopPropagation()
                      inputRef.current?.click()
                    }}
                    sx={{ mt: 1, textTransform: 'none' }}
                  >
                    Обрати фотографію
                  </Button>
                  <Typography variant="caption" color="text.secondary">JPEG або PNG · до 5 МБ</Typography>
                </Stack>
              )}
            </Box>

            {file && (
              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5} sx={{ mt: 2, justifyContent: 'space-between', alignItems: { xs: 'stretch', sm: 'center' } }}>
                <Box sx={{ minWidth: 0 }}>
                  <Typography sx={{ fontWeight: 700, overflowWrap: 'anywhere' }}>{file.name}</Typography>
                  <Typography variant="body2" color="text.secondary">{formatFileSize(file.size)}</Typography>
                </Box>
                <Stack direction="row" spacing={1}>
                  <Button variant="outlined" onClick={() => inputRef.current?.click()} disabled={loading} sx={{ textTransform: 'none' }}>
                    Спробувати інше
                  </Button>
                  <Button color="inherit" onClick={removeFile} disabled={loading} sx={{ textTransform: 'none' }}>
                    Видалити фото
                  </Button>
                </Stack>
              </Stack>
            )}

            {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}

            <Button
              variant="contained"
              size="large"
              fullWidth
              onClick={recognize}
              disabled={!file || loading}
              sx={{ mt: 3, py: 1.5, textTransform: 'none', fontSize: '1rem', fontWeight: 700 }}
              startIcon={loading ? <CircularProgress size={20} color="inherit" /> : undefined}
            >
              {loading ? 'Розпізнаємо знак…' : 'Розпізнати знак'}
            </Button>
          </CardContent>
        </Card>

        <Card variant="outlined" sx={{ minHeight: 420, boxShadow: '0 12px 36px rgba(31, 41, 55, 0.06)' }}>
          <CardContent sx={{ p: { xs: 2.5, sm: 3.5 } }}>
            {!result && !loading && (
              <Stack spacing={2} sx={{ minHeight: 350, alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
                <Box sx={{ width: 70, height: 70, borderRadius: 3, backgroundColor: 'secondary.main', opacity: 0.1 }} />
                <Typography variant="h6">Результат з’явиться тут</Typography>
                <Typography color="text.secondary" sx={{ maxWidth: 330 }}>
                  Завантажте фото та натисніть «Розпізнати знак».
                </Typography>
              </Stack>
            )}

            {loading && (
              <Stack spacing={2} sx={{ minHeight: 350, alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
                <CircularProgress size={52} />
                <Typography variant="h6">Аналізуємо зображення</Typography>
                <Typography color="text.secondary">Це може зайняти кілька секунд</Typography>
              </Stack>
            )}

            {result && !loading && (
              <Stack spacing={3}>
                <Box>
                  <Typography variant="overline" color="secondary.main" sx={{ fontWeight: 800, letterSpacing: 1.5 }}>
                    Основний результат
                  </Typography>
                  <Typography variant="h4" sx={{ mt: 1, mb: 1, fontSize: { xs: '1.65rem', md: '2rem' } }}>
                    {result.label}
                  </Typography>
                  <Stack direction="row" spacing={2} sx={{ alignItems: 'center' }}>
                    <Typography variant="h5" color="primary.main" sx={{ fontWeight: 800 }}>
                      {confidencePercent(result.confidence)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">ID класу: {result.classId}</Typography>
                  </Stack>
                </Box>

                <Box>
                  <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 800 }}>Top-3 predictions</Typography>
                  <Stack spacing={2}>
                    {result.topPredictions.map((item, index) => {
                      const percent = item.confidence <= 1 ? item.confidence * 100 : item.confidence
                      return (
                        <Box key={`${item.classId}-${index}`}>
                          <Stack direction="row" spacing={1} sx={{ mb: 0.75, justifyContent: 'space-between', alignItems: 'baseline' }}>
                            <Typography sx={{ fontWeight: index === 0 ? 800 : 600 }}>{item.label}</Typography>
                            <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: 'nowrap', ml: 1 }}>
                              {confidencePercent(item.confidence)}
                            </Typography>
                          </Stack>
                          <LinearProgress
                            variant="determinate"
                            value={Math.min(100, Math.max(0, percent))}
                            aria-label={`${item.label}: ${confidencePercent(item.confidence)}`}
                            sx={{ height: 8, borderRadius: 4, backgroundColor: 'grey.200', '& .MuiLinearProgress-bar': { borderRadius: 4 } }}
                          />
                        </Box>
                      )
                    })}
                  </Stack>
                </Box>

                <Alert severity="info" sx={{ backgroundColor: 'grey.50' }}>
                  Модель класифікує один дорожній знак на зображенні. Результат може бути неточним для складних фотографій
                </Alert>
              </Stack>
            )}
          </CardContent>
        </Card>
      </Box>
    </Container>
  )
}
