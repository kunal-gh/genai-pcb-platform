import React, { useState, useEffect } from 'react';
import {
  Container,
  Box,
  AppBar,
  Toolbar,
  Typography,
  Button,
  Grid,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import HistoryIcon from '@mui/icons-material/History';
import PromptInput from '../components/PromptInput';
import ProcessingStatus from '../components/ProcessingStatus';
import DesignPreview from '../components/DesignPreview';
import SchematicPreview from '../components/SchematicPreview';
import ErrorDisplay, { DesignError } from '../components/ErrorDisplay';
import FileDownloadManager from '../components/FileDownloadManager';
import { designApi, DesignStatus, DesignFile } from '../services/api';

const DesignPage: React.FC = () => {
  const navigate = useNavigate();
  const [processing, setProcessing] = useState(false);
  const [designId, setDesignId] = useState<string | null>(null);
  const [status, setStatus] = useState<DesignStatus | null>(null);
  const [files, setFiles] = useState<DesignFile[]>([]);
  const [errors, setErrors] = useState<DesignError[]>([]);
  const [schematicPreview, setSchematicPreview] = useState<string | undefined>();
  const [pcbPreview, setPcbPreview] = useState<string | undefined>();

  useEffect(() => {
    if (!designId) return;

    const pollStatus = async () => {
      try {
        const statusData = await designApi.getDesignStatus(designId);
        setStatus(statusData);

        if (statusData.status === 'completed') {
          const details = await designApi.getDesignDetails(designId);
          setFiles(details.files);
          setProcessing(false);

          // Load preview images
          const schematic = await designApi.getPreviewImage(designId, 'schematic');
          const pcb = await designApi.getPreviewImage(designId, 'pcb');
          setSchematicPreview(schematic || undefined);
          setPcbPreview(pcb || undefined);

          // Load verification results
          const verificationResults = await designApi.getVerificationResults(designId);
          const mappedErrors: DesignError[] = verificationResults.map(result => ({
            type: result.severity === 'CRITICAL' || result.severity === 'ERROR' ? 'error' : 
                  result.severity === 'WARNING' ? 'warning' : 'info',
            category: result.category,
            message: result.message,
            suggestion: result.suggestion,
            location: result.location,
          }));
          setErrors(mappedErrors);
        } else if (statusData.status === 'failed') {
          setProcessing(false);
        }
      } catch (error) {
        console.error('Error polling status:', error);
      }
    };

    const interval = setInterval(pollStatus, 2000);
    return () => clearInterval(interval);
  }, [designId]);

  const handleSubmit = async (prompt: string) => {
    try {
      setProcessing(true);
      setFiles([]);
      const response = await designApi.createDesign({ prompt });
      setDesignId(response.design_id);
    } catch (error) {
      console.error('Error creating design:', error);
      setProcessing(false);
    }
  };

  const handleDownload = async (fileId: string, filename: string) => {
    if (!designId) return;
    
    try {
      const blob = await designApi.downloadFile(designId, fileId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error downloading file:', error);
    }
  };

  const handleDownloadAll = async (fileIds: string[]) => {
    if (!designId) return;
    
    // Download multiple files as a zip (future enhancement)
    // For now, download them individually
    for (const fileId of fileIds) {
      const file = files.find(f => f.id === fileId);
      if (file) {
        await handleDownload(fileId, file.filename);
      }
    }
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            GenAI PCB Design Platform
          </Typography>
          <Button
            color="inherit"
            startIcon={<HistoryIcon />}
            onClick={() => navigate('/history')}
          >
            History
          </Button>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <PromptInput onSubmit={handleSubmit} disabled={processing} />
          </Grid>

          {status && (
            <Grid item xs={12}>
              <ProcessingStatus
                status={status.status}
                progress={status.progress}
                currentStep={status.current_step}
                errorMessage={status.error_message}
              />
            </Grid>
          )}

          {files.length > 0 && (
            <>
              <Grid item xs={12} md={6}>
                <SchematicPreview
                  imageUrl={schematicPreview}
                  title="Schematic Preview"
                />
              </Grid>

              <Grid item xs={12} md={6}>
                <SchematicPreview
                  imageUrl={pcbPreview}
                  title="PCB Layout Preview"
                />
              </Grid>

              <Grid item xs={12}>
                <FileDownloadManager
                  files={files}
                  onDownload={handleDownload}
                  onDownloadAll={handleDownloadAll}
                />
              </Grid>

              {errors.length > 0 && (
                <Grid item xs={12}>
                  <ErrorDisplay errors={errors} />
                </Grid>
              )}
            </>
          )}
        </Grid>
      </Container>
    </Box>
  );
};

export default DesignPage;
