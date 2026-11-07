import { useEffect, useState } from 'react'
import { BrowserRouter, Link, NavLink, Route, Routes, useNavigate } from 'react-router-dom'
import {
  Alert,
  AppBar,
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Container,
  CssBaseline,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Stack,
  Toolbar,
  Typography,
} from '@mui/material'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import { getHealthStatus } from './api/health'
import './App.css'

const navigation = [
  { label: 'Головна', path: '/' },
  { label: 'Розпізнавання', path: '/recognition' },
  { label: 'Каталог', path: '/catalog' },
  { label: 'Тести', path: '/tests' },
  { label: 'Історія', path: '/history' },
]

const theme = createTheme({
  palette: {
    primary: { main: '#2563eb' },
    secondary: { main: '#0f766e' },
    background: { default: '#f5f7fb', paper: '#ffffff' },
  },
  typography: {
    fontFamily: '"Inter", "Segoe UI", sans-serif',
    h1: { fontWeight: 800, letterSpacing: '-0.04em' },
    h2: { fontWeight: 750, letterSpacing: '-0.03em' },
  },
  shape: { borderRadius: 16 },
})

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <AppLayout />
      </BrowserRouter>
    </ThemeProvider>
  )
}

function AppLayout() {
  const [drawerOpen, setDrawerOpen] = useState(false)

  return (
    <Box sx={{ minHeight: '100vh' }}>
      <AppBar position="sticky" color="inherit" elevation={0} sx={{ borderBottom: '1px solid', borderColor: 'divider' }}>
        <Toolbar sx={{ justifyContent: 'space-between', py: 1 }}>
          <Button component={Link} to="/" color="inherit" sx={{ textTransform: 'none', p: 0 }}>
            <Typography variant="h6" sx={{ fontWeight: 800, color: 'primary.main' }}>
              DriveSmart <Box component="span" sx={{ color: 'secondary.main' }}>AI</Box>
            </Typography>
          </Button>
          <Box component="nav" sx={{ display: { xs: 'none', md: 'flex' }, gap: 1 }}>
            {navigation.map((item) => <NavigationLink key={item.path} {...item} />)}
          </Box>
          <IconButton aria-label="Відкрити меню" onClick={() => setDrawerOpen(true)} sx={{ display: { md: 'none' } }}>
            <Typography sx={{ fontSize: '1.4rem' }}>☰</Typography>
          </IconButton>
        </Toolbar>
      </AppBar>
      <Drawer anchor="right" open={drawerOpen} onClose={() => setDrawerOpen(false)}>
        <List sx={{ width: 260, pt: 2 }} aria-label="Основна навігація">
          {navigation.map((item) => (
            <ListItem key={item.path} disablePadding>
              <ListItemButton component={NavLink} to={item.path} onClick={() => setDrawerOpen(false)}>
                <ListItemText primary={item.label} />
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Drawer>
      <Box component="main">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/recognition" element={<RecognitionPage />} />
          <Route path="/catalog" element={<PlaceholderPage title="Каталог знаків" description="Тут з’явиться довідник із 43 класів дорожніх знаків." />} />
          <Route path="/tests" element={<PlaceholderPage title="Навчальні тести" description="Тут з’являться інтерактивні тести для перевірки знань дорожніх знаків." />} />
          <Route path="/history" element={<PlaceholderPage title="Історія розпізнавань" description="Історія стане доступною після підключення авторизації та backend API." />} />
          <Route path="*" element={<PlaceholderPage title="Сторінку не знайдено" description="Перевірте адресу або поверніться на головну." />} />
        </Routes>
      </Box>
    </Box>
  )
}

function NavigationLink({ label, path }: { label: string; path: string }) {
  return (
    <Button
      component={NavLink}
      to={path}
      sx={{
        color: 'text.secondary',
        textTransform: 'none',
        '&.active': { color: 'primary.main', backgroundColor: 'primary.50' },
      }}
    >
      {label}
    </Button>
  )
}

function HomePage() {
  const navigate = useNavigate()
  const [status, setStatus] = useState<'loading' | 'online' | 'offline'>('loading')

  useEffect(() => {
    getHealthStatus()
      .then(() => setStatus('online'))
      .catch(() => setStatus('offline'))
  }, [])

  return (
    <Container maxWidth="lg" sx={{ py: { xs: 6, md: 12 } }}>
      <Box className="hero" sx={{ p: { xs: 3, md: 7 }, mb: 5 }}>
        <Stack spacing={3} sx={{ maxWidth: 680 }}>
          <Typography variant="overline" sx={{ color: 'secondary.main', fontWeight: 800, letterSpacing: 2 }}>
            Розумніше за кермом
          </Typography>
          <Typography variant="h1" sx={{ fontSize: { xs: '2.7rem', md: '4.8rem' }, lineHeight: 1 }}>
            Розпізнавайте дорожні знаки впевнено
          </Typography>
          <Typography variant="h6" color="text.secondary" sx={{ maxWidth: 570, fontWeight: 400, lineHeight: 1.6 }}>
            DriveSmart AI допомагає визначити дорожній знак за фотографією та краще орієнтуватися на дорозі.
          </Typography>
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
            <Button variant="contained" size="large" onClick={() => navigate('/recognition')} sx={{ px: 3, py: 1.5 }}>
              Розпізнати знак
            </Button>
            <Button component={Link} to="/catalog" variant="outlined" size="large" sx={{ px: 3, py: 1.5 }}>
              Переглянути каталог
            </Button>
          </Stack>
          <BackendStatus status={status} />
        </Stack>
      </Box>
      <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
        <FeatureCard icon="01" title="Простий upload" text="Завантажте фотографію знака за кілька секунд." />
        <FeatureCard icon="02" title="Швидкий аналіз" text="Отримуйте зрозумілий результат без зайвих кроків." />
        <FeatureCard icon="03" title="Вчіться більше" text="Вивчайте знаки в каталозі та перевіряйте знання в тестах." />
      </Stack>
    </Container>
  )
}

function BackendStatus({ status }: { status: 'loading' | 'online' | 'offline' }) {
  if (status === 'loading') {
    return <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}><CircularProgress size={18} /><Typography color="text.secondary">Перевіряємо backend…</Typography></Stack>
  }
  return <Alert severity={status === 'online' ? 'success' : 'error'} sx={{ width: 'fit-content' }}>{status === 'online' ? 'Сервер працює' : 'Сервер недоступний'}</Alert>
}

function FeatureCard({ icon, title, text }: { icon: string; title: string; text: string }) {
  return <Card variant="outlined" sx={{ flex: 1, borderColor: 'transparent', boxShadow: '0 8px 30px rgba(31, 41, 55, 0.06)' }}><CardContent sx={{ p: 3 }}><Box sx={{ color: 'primary.main', mb: 2 }}>{icon}</Box><Typography variant="h6" gutterBottom>{title}</Typography><Typography color="text.secondary">{text}</Typography></CardContent></Card>
}

function RecognitionPage() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl) }, [previewUrl])

  const handleFile = (selectedFile: File | undefined) => {
    if (!selectedFile) return
    if (!selectedFile.type.startsWith('image/')) {
      setError('Будь ласка, оберіть файл зображення.')
      return
    }
    if (selectedFile.size > 10 * 1024 * 1024) {
      setError('Розмір зображення не може перевищувати 10 МБ.')
      return
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setFile(selectedFile)
    setPreviewUrl(URL.createObjectURL(selectedFile))
    setError(null)
  }

  return <Container maxWidth="md" sx={{ py: { xs: 5, md: 9 } }}>
    <Stack spacing={1} sx={{ mb: 4 }}><Typography variant="h2">Розпізнавання</Typography><Typography color="text.secondary">Завантажте фото одного дорожнього знака для підготовки до аналізу.</Typography></Stack>
    <Card variant="outlined"><CardContent sx={{ p: { xs: 2, md: 4 } }}>
      <Stack spacing={3} sx={{ alignItems: 'center' }}>
        <Button component="label" variant="contained" size="large">Обрати фотографію<input hidden type="file" accept="image/*" onChange={(event) => handleFile(event.target.files?.[0])} /></Button>
        <Typography variant="body2" color="text.secondary">PNG, JPEG або WEBP до 10 МБ</Typography>
        {error && <Alert severity="error" sx={{ width: '100%' }}>{error}</Alert>}
        {previewUrl && file && <Box sx={{ width: '100%', textAlign: 'center' }}><Box component="img" src={previewUrl} alt={`Попередній перегляд: ${file.name}`} sx={{ maxWidth: '100%', maxHeight: 420, borderRadius: 2, objectFit: 'contain' }} /><Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>{file.name}</Typography><Alert severity="info" sx={{ mt: 3, textAlign: 'left' }}>Реальне розпізнавання буде доступне після підключення ML-моделі.</Alert></Box>}
      </Stack>
    </CardContent></Card>
  </Container>
}

function PlaceholderPage({ title, description }: { title: string; description: string }) {
  return <Container maxWidth="md" sx={{ py: { xs: 7, md: 12 } }}><Card variant="outlined"><CardContent sx={{ p: { xs: 3, md: 6 }, textAlign: 'center' }}><Typography variant="h2" sx={{ mb: 2 }}>{title}</Typography><Typography color="text.secondary">{description}</Typography></CardContent></Card></Container>
}

export default App
