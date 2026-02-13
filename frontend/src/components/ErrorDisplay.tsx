import React from 'react';
import {
  Alert,
  AlertTitle,
  Box,
  Typography,
  List,
  ListItem,
  ListItemText,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import WarningIcon from '@mui/icons-material/Warning';
import InfoIcon from '@mui/icons-material/Info';

export interface DesignError {
  type: 'error' | 'warning' | 'info';
  category: string;
  message: string;
  suggestion?: string;
  location?: string;
}

interface ErrorDisplayProps {
  errors: DesignError[];
}

const ErrorDisplay: React.FC<ErrorDisplayProps> = ({ errors }) => {
  if (errors.length === 0) {
    return null;
  }

  const getIcon = (type: string) => {
    switch (type) {
      case 'error':
        return <ErrorOutlineIcon />;
      case 'warning':
        return <WarningIcon />;
      default:
        return <InfoIcon />;
    }
  };

  const getSeverity = (type: string): 'error' | 'warning' | 'info' => {
    return type as 'error' | 'warning' | 'info';
  };

  const groupedErrors = errors.reduce((acc, error) => {
    if (!acc[error.category]) {
      acc[error.category] = [];
    }
    acc[error.category].push(error);
    return acc;
  }, {} as Record<string, DesignError[]>);

  return (
    <Box sx={{ mt: 2 }}>
      {Object.entries(groupedErrors).map(([category, categoryErrors]) => (
        <Accordion key={category} defaultExpanded={categoryErrors.some(e => e.type === 'error')}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {getIcon(categoryErrors[0].type)}
              <Typography>
                {category} ({categoryErrors.length})
              </Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <List dense>
              {categoryErrors.map((error, index) => (
                <ListItem key={index} sx={{ flexDirection: 'column', alignItems: 'flex-start' }}>
                  <Alert severity={getSeverity(error.type)} sx={{ width: '100%', mb: 1 }}>
                    <AlertTitle>{error.message}</AlertTitle>
                    {error.location && (
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        Location: {error.location}
                      </Typography>
                    )}
                    {error.suggestion && (
                      <Typography variant="body2" color="text.secondary">
                        Suggestion: {error.suggestion}
                      </Typography>
                    )}
                  </Alert>
                </ListItem>
              ))}
            </List>
          </AccordionDetails>
        </Accordion>
      ))}
    </Box>
  );
};

export default ErrorDisplay;
