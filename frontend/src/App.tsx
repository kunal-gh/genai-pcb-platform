import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import DesignPage from './pages/DesignPage';
import HistoryPage from './pages/HistoryPage';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Routes>
          <Route path="/" element={<DesignPage />} />
          <Route path="/history" element={<HistoryPage />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;
