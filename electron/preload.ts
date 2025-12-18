import { contextBridge } from 'electron';

contextBridge.exposeInMainWorld('telemitry', {
  version: '0.1.0'
});
