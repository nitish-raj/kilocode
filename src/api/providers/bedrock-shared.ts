import { BedrockRuntimeClient, BedrockRuntimeClientConfig } from "@aws-sdk/client-bedrock-runtime"
import { fromIni } from "@aws-sdk/credential-providers"

/**
 * Options for creating a Bedrock Runtime Client
 */
export interface BedrockClientOptions {
	awsRegion?: string
	awsBedrockEndpoint?: string
	awsBedrockEndpointEnabled?: boolean
	awsAccessKey?: string
	awsSecretKey?: string
	awsSessionToken?: string
	awsProfile?: string
	awsUseProfile?: boolean
	awsApiKey?: string
	awsUseApiKey?: boolean
	awsCustomArn?: string
}

/**
 * Creates a configured BedrockRuntimeClient with the specified options
 * Handles region resolution, endpoint override, and various authentication methods
 * including direct credentials, profile-based credentials, and API key authentication
 *
 * @param options Configuration options for the Bedrock client
 * @returns Configured BedrockRuntimeClient instance
 */
export function createBedrockRuntimeClient(options: BedrockClientOptions): BedrockRuntimeClient {
	// Extended type to support custom authentication properties
	const clientConfig: BedrockRuntimeClientConfig & {
		token?: { token: string }
		authSchemePreference?: string[]
		maxAttempts?: number
	} = {
		region: options.awsRegion,
		// Add the endpoint configuration when specified and enabled
		...(options.awsBedrockEndpoint &&
			options.awsBedrockEndpointEnabled && { endpoint: options.awsBedrockEndpoint }),
		// Always set maxAttempts to 1 for consistency with original behavior
		maxAttempts: 1,
	}

	if (options.awsUseApiKey && options.awsApiKey) {
		// Use API key/token-based authentication if enabled and API key is set
		clientConfig.token = { token: options.awsApiKey }
		clientConfig.authSchemePreference = ["httpBearerAuth"] // Otherwise there's no end of credential problems.
		clientConfig.requestHandler = {
			// This should be the default anyway, but without setting something
			// this provider fails to work with LiteLLM passthrough.
			requestTimeout: 0,
		}
	} else if (options.awsUseProfile && options.awsProfile) {
		// Use profile-based credentials if enabled and profile is set
		clientConfig.credentials = fromIni({
			profile: options.awsProfile,
			ignoreCache: true,
		})
	} else if (options.awsAccessKey && options.awsSecretKey) {
		// Use direct credentials if provided
		clientConfig.credentials = {
			accessKeyId: options.awsAccessKey,
			secretAccessKey: options.awsSecretKey,
			...(options.awsSessionToken ? { sessionToken: options.awsSessionToken } : {}),
		}
	} else {
		// Create empty credentials for consistency with original embedding provider behavior
		clientConfig.credentials = {
			accessKeyId: "",
			secretAccessKey: "",
			sessionToken: undefined,
		}
	}

	return new BedrockRuntimeClient(clientConfig)
}
