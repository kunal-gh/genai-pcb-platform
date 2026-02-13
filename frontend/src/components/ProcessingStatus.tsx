import React from 'react';
import {
  Box,
  Paper,
  Typography,
  LinearProgress,
  Stepper,
  Step,
  StepLabel,
  Alert,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';

interface ProcessingStatusProps {
  status: string;
  progress: number;
  currentStep: string;
  errorMessage?: string;
}

const steps = [
  'Parsing prompt',
  'Generating SKiDL code',
  'Creating schematic',
  'Generating PCB layout',
  'Running verification',
  'Exporting files',
];

const ProcessingStatus: React.FC<ProcessingStatusProps> = ({
  status,
  progress,
  currentStep,
  errorMessage,
}) => {
  const getActiveStep = () => {
    const stepIndex = steps.findIndex(step => 
      currentStep.toLowerCase().includes(step.toLowerCase().split(' ')[0])
    );
    return stepIndex >= 0 ? stepIndex : 0;
  };

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Processing Status
      </Typography>

      {status === 'failed' && errorMessage && (
        <Alert severity="error" icon={<ErrorIcon />} sx={{ mb: 2 }}>
          {errorMessage}
        </Alert>
      )}

      {status === 'completed' && (
        <Alert severity="success" icon={<CheckCircleIcon />} sx={{ mb: 2 }}>
          Design completed successfully!
        </Alert>
      )}

      {status === 'processing' && (
        <>
          <Box sx={{ mb: 3 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              {currentStep}
            </Typography>
            <LinearProgress variant="determinate" value={progress} />
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
              {progress}% complete
            </Typography>
          </Box>

          <Stepper activeStep={getActiveStep()} alternativeLabel>
            {steps.map((label) => (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
              </Step>
            ))}
          </Stepper>
        </>
      )}
    </Paper>
  );
};

export default ProcessingStatus;
