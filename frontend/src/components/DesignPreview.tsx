import React from 'react';
import {
  Box,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Divider,
} from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import DescriptionIcon from '@mui/icons-material/Description';
import { DesignFile } from '../services/api';

interface DesignPreviewProps {
  files: DesignFile[];
  onDownload: (fileId: string, filename: string) => void;
}

const getFileIcon = (fileType: string) => {
  return <DescriptionIcon />;
};

const DesignPreview: React.FC<DesignPreviewProps> = ({ files, onDownload }) => {
  if (files.length === 0) {
    return null;
  }

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Design Files
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Download your generated design files below
      </Typography>

      <List>
        {files.map((file, index) => (
          <React.Fragment key={file.id}>
            {index > 0 && <Divider />}
            <ListItem
              secondaryAction={
                <IconButton
                  edge="end"
                  onClick={() => onDownload(file.id, file.filename)}
                  aria-label="download"
                >
                  <DownloadIcon />
                </IconButton>
              }
            >
              <ListItemIcon>
                {getFileIcon(file.file_type)}
              </ListItemIcon>
              <ListItemText
                primary={file.filename}
                secondary={file.file_type}
              />
            </ListItem>
          </React.Fragment>
        ))}
      </List>
    </Paper>
  );
};

export default DesignPreview;
