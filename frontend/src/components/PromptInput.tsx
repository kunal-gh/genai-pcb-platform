import React, { useState } from 'react';
import {
  Box,
  TextField,
  Button,
  Paper,
  Typography,
  Alert,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';

interface PromptInputProps {
  onSubmit: (prompt: string) => void;
  disabled?: boolean;
}

const PromptInput: React.FC<PromptInputProps> = ({ onSubmit, disabled }) => {
  const [prompt, setPrompt] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = () => {
    if (prompt.trim().length < 10) {
      setError('Prompt must be at least 10 characters long');
      return;
    }
    if (prompt.trim().length > 10000) {
      setError('Prompt must be less than 10,000 characters');
      return;
    }
    setError('');
    onSubmit(prompt.trim());
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      handleSubmit();
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Describe Your PCB Design
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Describe your circuit in natural language. Include components, connections, and any specific requirements.
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <TextField
        fullWidth
        multiline
        rows={6}
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyPress={handleKeyPress}
        placeholder="Example: Create a simple LED circuit with a 5V power supply, a 330 ohm resistor, and a red LED. The LED should be connected in series with the resistor."
        disabled={disabled}
        sx={{ mb: 2 }}
      />

      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="caption" color="text.secondary">
          {prompt.length} / 10,000 characters (Ctrl+Enter to submit)
        </Typography>
        <Button
          variant="contained"
          endIcon={<SendIcon />}
          onClick={handleSubmit}
          disabled={disabled || prompt.trim().length < 10}
        >
          Generate Design
        </Button>
      </Box>
    </Paper>
  );
};

export default PromptInput;
