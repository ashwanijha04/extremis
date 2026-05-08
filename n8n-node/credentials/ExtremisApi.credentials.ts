import { ICredentialType, INodeProperties } from 'n8n-workflow';

export class ExtremisApi implements ICredentialType {
  name = 'extremisApi';
  displayName = 'Extremis API';
  documentationUrl = 'https://ashwanijha04.github.io/extremis/docs/deployment/render/';
  properties: INodeProperties[] = [
    {
      displayName: 'Server URL',
      name: 'serverUrl',
      type: 'string',
      default: 'https://your-server.onrender.com',
      placeholder: 'https://your-extremis-server.onrender.com',
      description: 'URL of your extremis server. Deploy one free at render.com.',
    },
    {
      displayName: 'API Key',
      name: 'apiKey',
      type: 'string',
      typeOptions: { password: true },
      default: '',
      description: 'extremis_sk_... key from your server logs on first start.',
    },
  ];
}
