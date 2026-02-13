import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import DesignPreview from '../DesignPreview';
import { DesignFile } from '../../services/api';

describe('DesignPreview', () => {
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

  it('renders design files', () => {
    render(<DesignPreview files={mockFiles} onDownload={jest.fn()} />);
    
    expect(screen.getByText('Design Files')).toBeInTheDocument();
    expect(screen.getByText('schematic.kicad_sch')).toBeInTheDocument();
    expect(screen.getByText('pcb.kicad_pcb')).toBeInTheDocument();
  });

  it('calls onDownload when download button clicked', () => {
    const onDownload = jest.fn();
    render(<DesignPreview files={mockFiles} onDownload={onDownload} />);
    
    const downloadButtons = screen.getAllByLabelText('download');
    fireEvent.click(downloadButtons[0]);
    
    expect(onDownload).toHaveBeenCalledWith('1', 'schematic.kicad_sch');
  });

  it('renders nothing when no files', () => {
    const { container } = render(<DesignPreview files={[]} onDownload={jest.fn()} />);
    expect(container.firstChild).toBeNull();
  });
});
