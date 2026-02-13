import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import FileDownloadManager from '../FileDownloadManager';
import { DesignFile } from '../../services/api';

describe('FileDownloadManager', () => {
  const mockFiles: DesignFile[] = [
    {
      id: '1',
      filename: 'schematic.kicad_sch',
      file_type: 'schematic',
      file_path: '/path/to/schematic.kicad_sch',
    },
    {
      id: '2',
      filename: 'pcb.kicad_pcb',
      file_type: 'pcb',
      file_path: '/path/to/pcb.kicad_pcb',
    },
  ];

  it('renders file list with categories', () => {
    render(
      <FileDownloadManager
        files={mockFiles}
        onDownload={jest.fn()}
        onDownloadAll={jest.fn()}
      />
    );
    
    expect(screen.getByText('Download Files')).toBeInTheDocument();
    expect(screen.getByText('schematic.kicad_sch')).toBeInTheDocument();
    expect(screen.getByText('pcb.kicad_pcb')).toBeInTheDocument();
  });

  it('handles file selection', () => {
    render(
      <FileDownloadManager
        files={mockFiles}
        onDownload={jest.fn()}
        onDownloadAll={jest.fn()}
      />
    );
    
    const checkboxes = screen.getAllByRole('checkbox');
    fireEvent.click(checkboxes[0]);
    
    expect(checkboxes[0]).toBeChecked();
  });

  it('calls onDownload for single file', () => {
    const onDownload = jest.fn();
    render(
      <FileDownloadManager
        files={mockFiles}
        onDownload={onDownload}
        onDownloadAll={jest.fn()}
      />
    );
    
    const checkboxes = screen.getAllByRole('checkbox');
    fireEvent.click(checkboxes[0]);
    
    const downloadButton = screen.getByText(/Download \(1\)/);
    fireEvent.click(downloadButton);
    
    expect(onDownload).toHaveBeenCalledWith('1', 'schematic.kicad_sch');
  });

  it('calls onDownloadAll for multiple files', () => {
    const onDownloadAll = jest.fn();
    render(
      <FileDownloadManager
        files={mockFiles}
        onDownload={jest.fn()}
        onDownloadAll={onDownloadAll}
      />
    );
    
    fireEvent.click(screen.getByText('Select All'));
    fireEvent.click(screen.getByText(/Download \(2\)/));
    
    expect(onDownloadAll).toHaveBeenCalledWith(['1', '2']);
  });
});
