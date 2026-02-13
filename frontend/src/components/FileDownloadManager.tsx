import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Checkbox,
  Divider,
} from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import FolderZipIcon from '@mui/icons-material/FolderZip';
import { DesignFile } from '../services/api';

interface FileDownloadManagerProps {
  files: DesignFile[];
  onDownload: (fileId: string, filename: string) => void;
  onDownloadAll: (fileIds: string[]) => void;
}

const FileDownloadManager: React.FC<FileDownloadManagerProps> = ({
  files,
  onDownload,
  onDownloadAll,
}) => {
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);

  const handleToggle = (fileId: string) => {
    setSelectedFiles(prev =>
      prev.includes(fileId)
        ? prev.filter(id => id !== fileId)
        : [...prev, fileId]
    );
  };

  const handleSelectAll = () => {
    if (selectedFiles.length === files.length) {
      setSelectedFiles([]);
    } else {
      setSelectedFiles(files.map(f => f.id));
    }
  };

  const handleDownloadSelected = () => {
    if (selectedFiles.length === 1) {
      const file = files.find(f => f.id === selectedFiles[0]);
      if (file) {
        onDownload(file.id, file.filename);
      }
    } else {
      onDownloadAll(selectedFiles);
    }
  };

  const getFileCategory = (fileType: string): string => {
    const categories: Record<string, string> = {
      'schematic': 'Schematic Files',
      'pcb': 'PCB Layout Files',
      'gerber': 'Manufacturing Files',
      'bom': 'Bill of Materials',
      'netlist': 'Netlist Files',
      'simulation': 'Simulation Results',
    };
    return categories[fileType] || 'Other Files';
  };

  const groupedFiles = files.reduce((acc, file) => {
    const category = getFileCategory(file.file_type);
    if (!acc[category]) {
      acc[category] = [];
    }
    acc[category].push(file);
    return acc;
  }, {} as Record<string, DesignFile[]>);

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">
          Download Files
        </Typography>
        <Box>
          <Button
            size="small"
            onClick={handleSelectAll}
            sx={{ mr: 1 }}
          >
            {selectedFiles.length === files.length ? 'Deselect All' : 'Select All'}
          </Button>
          <Button
            variant="contained"
            startIcon={selectedFiles.length > 1 ? <FolderZipIcon /> : <DownloadIcon />}
            onClick={handleDownloadSelected}
            disabled={selectedFiles.length === 0}
          >
            Download {selectedFiles.length > 0 && `(${selectedFiles.length})`}
          </Button>
        </Box>
      </Box>

      {Object.entries(groupedFiles).map(([category, categoryFiles]) => (
        <Box key={category} sx={{ mb: 2 }}>
          <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
            {category}
          </Typography>
          <List dense>
            {categoryFiles.map((file, index) => (
              <React.Fragment key={file.id}>
                {index > 0 && <Divider />}
                <ListItem
                  button
                  onClick={() => handleToggle(file.id)}
                >
                  <ListItemIcon>
                    <Checkbox
                      edge="start"
                      checked={selectedFiles.includes(file.id)}
                      tabIndex={-1}
                      disableRipple
                    />
                  </ListItemIcon>
                  <ListItemText
                    primary={file.filename}
                    secondary={file.file_type}
                  />
                </ListItem>
              </React.Fragment>
            ))}
          </List>
        </Box>
      ))}
    </Paper>
  );
};

export default FileDownloadManager;
