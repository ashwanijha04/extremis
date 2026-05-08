import {
  IExecuteFunctions,
  INodeExecutionData,
  INodeType,
  INodeTypeDescription,
  NodeOperationError,
} from 'n8n-workflow';

export class Extremis implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'Extremis Memory',
    name: 'extremis',
    icon: 'file:extremis.svg',
    group: ['transform'],
    version: 1,
    subtitle: '={{$parameter["operation"]}}',
    description: 'Store, recall, and manage AI agent memory with extremis',
    defaults: { name: 'Extremis Memory' },
    inputs: ['main'],
    outputs: ['main'],
    credentials: [{ name: 'extremisApi', required: true }],
    properties: [
      {
        displayName: 'Operation',
        name: 'operation',
        type: 'options',
        noDataExpression: true,
        options: [
          { name: 'Remember', value: 'remember', description: 'Store a memory' },
          { name: 'Recall', value: 'recall', description: 'Search memories by query' },
          { name: 'Remember Now', value: 'rememberNow', description: 'Write directly to a memory layer' },
          { name: 'Report Outcome', value: 'reportOutcome', description: 'Give +1/-1 feedback on recalled memories' },
          { name: 'KG: Add Entity', value: 'kgAddEntity', description: 'Add an entity to the knowledge graph' },
          { name: 'KG: Add Relationship', value: 'kgAddRelationship', description: 'Connect two entities' },
          { name: 'KG: Query', value: 'kgQuery', description: 'Look up an entity and its connections' },
          { name: 'Consolidate', value: 'consolidate', description: 'Distil conversation logs into structured memories' },
        ],
        default: 'remember',
      },

      // ── remember ──────────────────────────────────────────────────────────
      {
        displayName: 'Content',
        name: 'content',
        type: 'string',
        typeOptions: { rows: 3 },
        default: '',
        required: true,
        displayOptions: { show: { operation: ['remember', 'rememberNow'] } },
        description: 'The text to remember',
      },
      {
        displayName: 'Conversation ID',
        name: 'conversationId',
        type: 'string',
        default: 'default',
        displayOptions: { show: { operation: ['remember'] } },
        description: 'Groups messages together for consolidation',
      },
      {
        displayName: 'Role',
        name: 'role',
        type: 'options',
        options: [
          { name: 'User', value: 'user' },
          { name: 'Assistant', value: 'assistant' },
          { name: 'System', value: 'system' },
        ],
        default: 'user',
        displayOptions: { show: { operation: ['remember'] } },
      },

      // ── rememberNow ────────────────────────────────────────────────────────
      {
        displayName: 'Layer',
        name: 'layer',
        type: 'options',
        options: [
          { name: 'Semantic (durable facts)', value: 'semantic' },
          { name: 'Procedural (behavioural rules)', value: 'procedural' },
          { name: 'Episodic (timestamped events)', value: 'episodic' },
          { name: 'Identity (who the user is)', value: 'identity' },
          { name: 'Working (temporary)', value: 'working' },
        ],
        default: 'semantic',
        displayOptions: { show: { operation: ['rememberNow'] } },
      },
      {
        displayName: 'Confidence',
        name: 'confidence',
        type: 'number',
        typeOptions: { minValue: 0, maxValue: 1 },
        default: 0.9,
        displayOptions: { show: { operation: ['rememberNow'] } },
      },

      // ── recall ─────────────────────────────────────────────────────────────
      {
        displayName: 'Query',
        name: 'query',
        type: 'string',
        default: '',
        required: true,
        displayOptions: { show: { operation: ['recall'] } },
        description: 'What to search for — a question, topic, or message',
      },
      {
        displayName: 'Limit',
        name: 'limit',
        type: 'number',
        default: 5,
        displayOptions: { show: { operation: ['recall'] } },
      },

      // ── reportOutcome ──────────────────────────────────────────────────────
      {
        displayName: 'Memory IDs',
        name: 'memoryIds',
        type: 'string',
        default: '',
        required: true,
        displayOptions: { show: { operation: ['reportOutcome'] } },
        description: 'Comma-separated memory IDs from a previous Recall result',
      },
      {
        displayName: 'Success',
        name: 'success',
        type: 'boolean',
        default: true,
        displayOptions: { show: { operation: ['reportOutcome'] } },
        description: 'true = helpful (+1), false = not helpful (-1)',
      },

      // ── KG ─────────────────────────────────────────────────────────────────
      {
        displayName: 'Entity Name',
        name: 'entityName',
        type: 'string',
        default: '',
        required: true,
        displayOptions: { show: { operation: ['kgAddEntity', 'kgAddRelationship', 'kgQuery'] } },
      },
      {
        displayName: 'Entity Type',
        name: 'entityType',
        type: 'options',
        options: [
          { name: 'Person', value: 'person' },
          { name: 'Organisation', value: 'org' },
          { name: 'Project', value: 'project' },
          { name: 'Group', value: 'group' },
          { name: 'Concept', value: 'concept' },
        ],
        default: 'person',
        displayOptions: { show: { operation: ['kgAddEntity'] } },
      },
      {
        displayName: 'Target Entity',
        name: 'toEntity',
        type: 'string',
        default: '',
        required: true,
        displayOptions: { show: { operation: ['kgAddRelationship'] } },
      },
      {
        displayName: 'Relationship',
        name: 'relType',
        type: 'string',
        default: 'knows',
        required: true,
        displayOptions: { show: { operation: ['kgAddRelationship'] } },
        placeholder: 'works_at, friend, building, owns...',
      },
    ],
  };

  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const credentials = await this.getCredentials('extremisApi');
    const baseUrl = (credentials.serverUrl as string).replace(/\/$/, '');
    const apiKey = credentials.apiKey as string;
    const operation = this.getNodeParameter('operation', 0) as string;

    const headers = {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    };

    const request = async (method: string, path: string, body?: object) => {
      const res = await this.helpers.httpRequest({
        method: method as any,
        url: `${baseUrl}${path}`,
        headers,
        body: body ? JSON.stringify(body) : undefined,
        returnFullResponse: false,
      });
      return res;
    };

    const items = this.getInputData();
    const results: INodeExecutionData[] = [];

    for (let i = 0; i < items.length; i++) {
      let data: any;

      try {
        if (operation === 'remember') {
          await request('POST', '/v1/memories/remember', {
            content: this.getNodeParameter('content', i),
            role: this.getNodeParameter('role', i),
            conversation_id: this.getNodeParameter('conversationId', i),
          });
          data = { success: true, operation: 'remember' };

        } else if (operation === 'recall') {
          data = await request('POST', '/v1/memories/recall', {
            query: this.getNodeParameter('query', i),
            limit: this.getNodeParameter('limit', i),
          });

        } else if (operation === 'rememberNow') {
          data = await request('POST', '/v1/memories/store', {
            content: this.getNodeParameter('content', i),
            layer: this.getNodeParameter('layer', i),
            confidence: this.getNodeParameter('confidence', i),
          });

        } else if (operation === 'reportOutcome') {
          const ids = (this.getNodeParameter('memoryIds', i) as string)
            .split(',').map((s: string) => s.trim()).filter(Boolean);
          await request('POST', '/v1/memories/report', {
            memory_ids: ids,
            success: this.getNodeParameter('success', i),
          });
          data = { success: true, operation: 'reportOutcome' };

        } else if (operation === 'kgAddEntity') {
          data = await request('POST', '/v1/kg/write', {
            operation: 'add_entity',
            name: this.getNodeParameter('entityName', i),
            entity_type: this.getNodeParameter('entityType', i),
          });

        } else if (operation === 'kgAddRelationship') {
          data = await request('POST', '/v1/kg/write', {
            operation: 'add_relationship',
            from_entity: this.getNodeParameter('entityName', i),
            to_entity: this.getNodeParameter('toEntity', i),
            rel_type: this.getNodeParameter('relType', i),
          });

        } else if (operation === 'kgQuery') {
          data = await request('POST', '/v1/kg/query', {
            name: this.getNodeParameter('entityName', i),
          });

        } else if (operation === 'consolidate') {
          data = await request('POST', '/v1/memories/consolidate', {});
        }

        results.push({ json: data ?? {} });
      } catch (error) {
        if (this.continueOnFail()) {
          results.push({ json: { error: (error as Error).message } });
        } else {
          throw new NodeOperationError(this.getNode(), error as Error, { itemIndex: i });
        }
      }
    }

    return [results];
  }
}
