import React, { useState, useEffect } from 'react';
import {
  Container,
  Box,
  AppBar,
  Toolbar,
  Typography,
  Button,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Chip,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import DeleteIcon from '@mui/icons-material/Delete';
import VisibilityIcon from '@mui/icons-material/Visibility';
import { designApi, DesignDetails } from '../services/api';

const HistoryPage: React.FC = () => {
  const navigate = useNavigate();
  const [designs, setDesigns] = useState<DesignDetails[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDesigns();
  }, []);

  const loadDesigns = async () => {
    try {
      const data = await designApi.listDesigns();
      setDesigns(data);
    } catch (error) {
      console.error('Error loading designs:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (designId: string) => {
    if (!window.confirm('Are you sure you want to delete this design?')) {
      return;
    }

    try {
      await designApi.deleteDesign(designId);
      setDesigns(designs.filter(d => d.id !== designId));
    } catch (error) {
      console.error('Error deleting design:', error);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'processing':
        return 'info';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static">
        <Toolbar>
          <IconButton
            edge="start"
            color="inherit"
            onClick={() => navigate('/')}
            sx={{ mr: 2 }}
          >
            <ArrowBackIcon />
          </IconButton>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Design History
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Prompt</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Created</TableCell>
                <TableCell>Files</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {designs.map((design) => (
                <TableRow key={design.id}>
                  <TableCell>
                    {design.prompt.substring(0, 100)}
                    {design.prompt.length > 100 && '...'}
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={design.status}
                      color={getStatusColor(design.status)}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    {new Date(design.created_at).toLocaleString()}
                  </TableCell>
                  <TableCell>{design.files.length}</TableCell>
                  <TableCell align="right">
                    <IconButton
                      size="small"
                      onClick={() => navigate(`/?design=${design.id}`)}
                    >
                      <VisibilityIcon />
                    </IconButton>
                    <IconButton
                      size="small"
                      onClick={() => handleDelete(design.id)}
                    >
                      <DeleteIcon />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))}
              {designs.length === 0 && !loading && (
                <TableRow>
                  <TableCell colSpan={5} align="center">
                    No designs yet. Create your first design!
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Container>
    </Box>
  );
};

export default HistoryPage;
