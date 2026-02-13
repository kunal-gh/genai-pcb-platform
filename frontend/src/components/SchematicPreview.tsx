import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  IconButton,
  Toolbar,
} from '@mui/material';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
import FitScreenIcon from '@mui/icons-material/FitScreen';

interface SchematicPreviewProps {
  imageUrl?: string;
  title: string;
}

const SchematicPreview: React.FC<SchematicPreviewProps> = ({ imageUrl, title }) => {
  const [zoom, setZoom] = useState(1);

  const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.25, 3));
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.25, 0.5));
  const handleFitScreen = () => setZoom(1);

  if (!imageUrl) {
    return (
      <Paper elevation={3} sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          {title}
        </Typography>
        <Box
          sx={{
            height: 400,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: 'grey.100',
            borderRadius: 1,
          }}
        >
          <Typography color="text.secondary">
            Preview not available
          </Typography>
        </Box>
      </Paper>
    );
  }

  return (
    <Paper elevation={3}>
      <Toolbar>
        <Typography variant="h6" sx={{ flexGrow: 1 }}>
          {title}
        </Typography>
        <IconButton onClick={handleZoomOut} disabled={zoom <= 0.5}>
          <ZoomOutIcon />
        </IconButton>
        <Typography variant="body2" sx={{ mx: 1 }}>
          {Math.round(zoom * 100)}%
        </Typography>
        <IconButton onClick={handleZoomIn} disabled={zoom >= 3}>
          <ZoomInIcon />
        </IconButton>
        <IconButton onClick={handleFitScreen}>
          <FitScreenIcon />
        </IconButton>
      </Toolbar>
      <Box
        sx={{
          height: 500,
          overflow: 'auto',
          bgcolor: 'grey.100',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          p: 2,
        }}
      >
        <img
          src={imageUrl}
          alt={title}
          style={{
            transform: `scale(${zoom})`,
            transformOrigin: 'center',
            maxWidth: '100%',
            transition: 'transform 0.2s',
          }}
        />
      </Box>
    </Paper>
  );
};

export default SchematicPreview;
