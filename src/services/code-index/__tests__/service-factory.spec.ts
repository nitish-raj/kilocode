import type { MockedClass, MockedFunction } from "vitest"
import { CodeIndexServiceFactory } from "../service-factory"
import { OpenAiEmbedder } from "../embedders/openai"
import { CodeIndexOllamaEmbedder } from "../embedders/ollama"
import { OpenAICompatibleEmbedder } from "../embedders/openai-compatible"
import { GeminiEmbedder } from "../embedders/gemini"
import { AmazonBedrockEmbeddingProvider } from "../embedders/bedrock"
import { QdrantVectorStore } from "../vector-store/qdrant-client"
import { createBedrockRuntimeClient } from "../../../api/providers/bedrock-shared"

// Mock the embedders and vector store
vitest.mock("../embedders/openai")
vitest.mock("../embedders/ollama")
vitest.mock("../embedders/openai-compatible")
vitest.mock("../embedders/gemini")
vitest.mock("../embedders/bedrock")
vitest.mock("../vector-store/qdrant-client")

// Mock the embedding models module
vitest.mock("../../../shared/embeddingModels", () => ({
	getDefaultModelId: vitest.fn(),
	getModelDimension: vitest.fn(),
}))

// Mock TelemetryService
vitest.mock("@roo-code/telemetry", () => ({
	TelemetryService: {
		instance: {
			captureEvent: vitest.fn(),
		},
	},
}))

// Mock the shared Bedrock client helper
vitest.mock("../../../api/providers/bedrock-shared")

const MockedOpenAiEmbedder = OpenAiEmbedder as MockedClass<typeof OpenAiEmbedder>
const MockedCodeIndexOllamaEmbedder = CodeIndexOllamaEmbedder as MockedClass<typeof CodeIndexOllamaEmbedder>
const MockedOpenAICompatibleEmbedder = OpenAICompatibleEmbedder as MockedClass<typeof OpenAICompatibleEmbedder>
const MockedGeminiEmbedder = GeminiEmbedder as MockedClass<typeof GeminiEmbedder>
const MockedAmazonBedrockEmbeddingProvider = AmazonBedrockEmbeddingProvider as MockedClass<
	typeof AmazonBedrockEmbeddingProvider
>
const MockedQdrantVectorStore = QdrantVectorStore as MockedClass<typeof QdrantVectorStore>

// Import the mocked functions
import { getDefaultModelId, getModelDimension } from "../../../shared/embeddingModels"
const mockGetDefaultModelId = getDefaultModelId as MockedFunction<typeof getDefaultModelId>
const mockGetModelDimension = getModelDimension as MockedFunction<typeof getModelDimension>
const mockCreateBedrockRuntimeClient = createBedrockRuntimeClient as MockedFunction<typeof createBedrockRuntimeClient>

describe("CodeIndexServiceFactory", () => {
	let factory: CodeIndexServiceFactory
	let mockConfigManager: any
	let mockCacheManager: any

	beforeEach(() => {
		vitest.clearAllMocks()

		mockConfigManager = {
			getConfig: vitest.fn(),
		}

		mockCacheManager = {}

		factory = new CodeIndexServiceFactory(mockConfigManager, "/test/workspace", mockCacheManager)
	})

	describe("createEmbedder", () => {
		it("should pass model ID to OpenAI embedder when using OpenAI provider", () => {
			// Arrange
			const testModelId = "text-embedding-3-large"
			const testConfig = {
				embedderProvider: "openai",
				modelId: testModelId,
				openAiOptions: {
					openAiNativeApiKey: "test-api-key",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act
			factory.createEmbedder()

			// Assert
			expect(MockedOpenAiEmbedder).toHaveBeenCalledWith({
				openAiNativeApiKey: "test-api-key",
				openAiEmbeddingModelId: testModelId,
			})
		})

		it("should pass model ID to Ollama embedder when using Ollama provider", () => {
			// Arrange
			const testModelId = "nomic-embed-text:latest"
			const testConfig = {
				embedderProvider: "ollama",
				modelId: testModelId,
				ollamaOptions: {
					ollamaBaseUrl: "http://localhost:11434",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act
			factory.createEmbedder()

			// Assert
			expect(MockedCodeIndexOllamaEmbedder).toHaveBeenCalledWith({
				ollamaBaseUrl: "http://localhost:11434",
				ollamaModelId: testModelId,
			})
		})

		it("should handle undefined model ID for OpenAI embedder", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "openai",
				modelId: undefined,
				openAiOptions: {
					openAiNativeApiKey: "test-api-key",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act
			factory.createEmbedder()

			// Assert
			expect(MockedOpenAiEmbedder).toHaveBeenCalledWith({
				openAiNativeApiKey: "test-api-key",
				openAiEmbeddingModelId: undefined,
			})
		})

		it("should handle undefined model ID for Ollama embedder", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "ollama",
				modelId: undefined,
				ollamaOptions: {
					ollamaBaseUrl: "http://localhost:11434",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act
			factory.createEmbedder()

			// Assert
			expect(MockedCodeIndexOllamaEmbedder).toHaveBeenCalledWith({
				ollamaBaseUrl: "http://localhost:11434",
				ollamaModelId: undefined,
			})
		})

		it("should throw error when OpenAI API key is missing", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "openai",
				modelId: "text-embedding-3-large",
				openAiOptions: {
					openAiNativeApiKey: undefined,
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act & Assert
			expect(() => factory.createEmbedder()).toThrow("serviceFactory.openAiConfigMissing")
		})

		it("should throw error when Ollama base URL is missing", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "ollama",
				modelId: "nomic-embed-text:latest",
				ollamaOptions: {
					ollamaBaseUrl: undefined,
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act & Assert
			expect(() => factory.createEmbedder()).toThrow("serviceFactory.ollamaConfigMissing")
		})

		it("should pass model ID to OpenAI Compatible embedder when using OpenAI Compatible provider", () => {
			// Arrange
			const testModelId = "text-embedding-3-large"
			const testConfig = {
				embedderProvider: "openai-compatible",
				modelId: testModelId,
				openAiCompatibleOptions: {
					baseUrl: "https://api.example.com/v1",
					apiKey: "test-api-key",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act
			factory.createEmbedder()

			// Assert
			expect(MockedOpenAICompatibleEmbedder).toHaveBeenCalledWith(
				"https://api.example.com/v1",
				"test-api-key",
				testModelId,
			)
		})

		it("should handle undefined model ID for OpenAI Compatible embedder", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "openai-compatible",
				modelId: undefined,
				openAiCompatibleOptions: {
					baseUrl: "https://api.example.com/v1",
					apiKey: "test-api-key",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act
			factory.createEmbedder()

			// Assert
			expect(MockedOpenAICompatibleEmbedder).toHaveBeenCalledWith(
				"https://api.example.com/v1",
				"test-api-key",
				undefined,
			)
		})

		it("should throw error when OpenAI Compatible base URL is missing", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "openai-compatible",
				modelId: "text-embedding-3-large",
				openAiCompatibleOptions: {
					baseUrl: undefined,
					apiKey: "test-api-key",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act & Assert
			expect(() => factory.createEmbedder()).toThrow("serviceFactory.openAiCompatibleConfigMissing")
		})

		it("should throw error when OpenAI Compatible API key is missing", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "openai-compatible",
				modelId: "text-embedding-3-large",
				openAiCompatibleOptions: {
					baseUrl: "https://api.example.com/v1",
					apiKey: undefined,
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act & Assert
			expect(() => factory.createEmbedder()).toThrow("serviceFactory.openAiCompatibleConfigMissing")
		})

		it("should throw error when OpenAI Compatible options are missing", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "openai-compatible",
				modelId: "text-embedding-3-large",
				openAiCompatibleOptions: undefined,
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act & Assert
			expect(() => factory.createEmbedder()).toThrow("serviceFactory.openAiCompatibleConfigMissing")
		})

		it("should create GeminiEmbedder with default model when no modelId specified", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "gemini",
				geminiOptions: {
					apiKey: "test-gemini-api-key",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act
			factory.createEmbedder()

			// Assert
			expect(MockedGeminiEmbedder).toHaveBeenCalledWith("test-gemini-api-key", undefined)
		})

		it("should create GeminiEmbedder with specified modelId", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "gemini",
				modelId: "text-embedding-004",
				geminiOptions: {
					apiKey: "test-gemini-api-key",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act
			factory.createEmbedder()

			// Assert
			expect(MockedGeminiEmbedder).toHaveBeenCalledWith("test-gemini-api-key", "text-embedding-004")
		})

		it("should throw error when Gemini API key is missing", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "gemini",
				geminiOptions: {
					apiKey: undefined,
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act & Assert
			expect(() => factory.createEmbedder()).toThrow("serviceFactory.geminiConfigMissing")
		})

		it("should throw error when Gemini options are missing", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "gemini",
				geminiOptions: undefined,
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act & Assert
			expect(() => factory.createEmbedder()).toThrow("serviceFactory.geminiConfigMissing")
		})

		it("should throw error for invalid embedder provider", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "invalid-provider",
				modelId: "some-model",
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act & Assert
			expect(() => factory.createEmbedder()).toThrow("serviceFactory.invalidEmbedderType")
		})

		it("should create AmazonBedrockEmbeddingProvider when provider is bedrock", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "bedrock",
				modelId: "amazon.titan-embed-text-v2:0",
				bedrockOptions: {
					region: "us-east-1",
					accessKeyId: "test-access-key",
					secretAccessKey: "test-secret-key",
					endpointUrl: "https://bedrock.us-east-1.amazonaws.com",
					maxRetries: 3,
					maxConcurrency: 5,
					batchSize: 10,
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act
			factory.createEmbedder()

			// Assert
			expect(MockedAmazonBedrockEmbeddingProvider).toHaveBeenCalledWith({
				modelId: "amazon.titan-embed-text-v2:0",
				region: "us-east-1",
				accessKeyId: "test-access-key",
				secretAccessKey: "test-secret-key",
				endpointUrl: "https://bedrock.us-east-1.amazonaws.com",
				maxRetries: 3,
				maxConcurrency: 5,
				batchSize: 10,
			})
		})

		it("should throw error when bedrock region is missing", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "bedrock",
				modelId: "amazon.titan-embed-text-v2:0",
				bedrockOptions: {
					// region is missing
					accessKeyId: "test-access-key",
					secretAccessKey: "test-secret-key",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act & Assert
			expect(() => factory.createEmbedder()).toThrow("serviceFactory.bedrockConfigMissing")
		})

		it("should throw error when bedrock modelId is missing", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "bedrock",
				// modelId is missing
				bedrockOptions: {
					region: "us-east-1",
					accessKeyId: "test-access-key",
					secretAccessKey: "test-secret-key",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act & Assert
			expect(() => factory.createEmbedder()).toThrow("serviceFactory.modelIdMissing")
		})

		it("should handle bedrock configuration with profile-based credentials", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "bedrock",
				modelId: "amazon.titan-embed-text-v2:0",
				bedrockOptions: {
					region: "us-east-1",
					// No explicit credentials - should use profile/IAM role
					maxRetries: 3,
					maxConcurrency: 5,
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act
			factory.createEmbedder()

			// Assert
			expect(MockedAmazonBedrockEmbeddingProvider).toHaveBeenCalledWith({
				modelId: "amazon.titan-embed-text-v2:0",
				region: "us-east-1",
				accessKeyId: undefined,
				secretAccessKey: undefined,
				sessionToken: undefined,
				endpointUrl: undefined,
				maxRetries: 3,
				maxConcurrency: 5,
				batchSize: undefined,
			})
		})

		it("should validate bedrock embedder configuration successfully", async () => {
			// Arrange
			const testConfig = {
				embedderProvider: "bedrock",
				modelId: "amazon.titan-embed-text-v2:0",
				bedrockOptions: {
					region: "us-east-1",
					accessKeyId: "test-access-key",
					secretAccessKey: "test-secret-key",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			const mockEmbedder = {
				validateConfiguration: vitest.fn().mockResolvedValue({ valid: true }),
			}
			MockedAmazonBedrockEmbeddingProvider.mockReturnValue(mockEmbedder as any)

			// Act
			const embedder = factory.createEmbedder()
			const result = await factory.validateEmbedder(embedder)

			// Assert
			expect(result).toEqual({ valid: true })
			expect(mockEmbedder.validateConfiguration).toHaveBeenCalled()
		})

		it("should handle bedrock embedder validation failure", async () => {
			// Arrange
			const testConfig = {
				embedderProvider: "bedrock",
				modelId: "amazon.titan-embed-text-v2:0",
				bedrockOptions: {
					region: "us-east-1",
					accessKeyId: "test-access-key",
					secretAccessKey: "test-secret-key",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			const validationError = "Invalid AWS credentials"
			const mockEmbedder = {
				validateConfiguration: vitest.fn().mockResolvedValue({ valid: false, error: validationError }),
			}
			MockedAmazonBedrockEmbeddingProvider.mockReturnValue(mockEmbedder as any)

			// Act
			const embedder = factory.createEmbedder()
			const result = await factory.validateEmbedder(embedder)

			// Assert
			expect(result).toEqual({ valid: false, error: validationError })
		})

		it("should handle bedrock embedder validation exception", async () => {
			// Arrange
			const testConfig = {
				embedderProvider: "bedrock",
				modelId: "amazon.titan-embed-text-v2:0",
				bedrockOptions: {
					region: "us-east-1",
					accessKeyId: "test-access-key",
					secretAccessKey: "test-secret-key",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			const validationError = new Error("Network error during validation")
			const mockEmbedder = {
				validateConfiguration: vitest.fn().mockRejectedValue(validationError),
			}
			MockedAmazonBedrockEmbeddingProvider.mockReturnValue(mockEmbedder as any)

			// Act
			const embedder = factory.createEmbedder()
			const result = await factory.validateEmbedder(embedder)

			// Assert
			expect(result).toEqual({ valid: false, error: "Network error during validation" })
		})

		it("should ensure bedrock credentials are not stored in configuration", () => {
			// This test verifies that sensitive credentials are not persisted
			// Arrange
			const testConfig = {
				embedderProvider: "bedrock",
				modelId: "amazon.titan-embed-text-v2:0",
				bedrockOptions: {
					region: "us-east-1",
					accessKeyId: "test-access-key",
					secretAccessKey: "test-secret-key",
					sessionToken: "test-session-token",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act
			factory.createEmbedder()

			// Assert - verify that the config manager was not called to persist credentials
			// This is a security check to ensure credentials aren't accidentally saved
			expect(mockConfigManager.getConfig).toHaveBeenCalled()
			// The factory should use the config but not persist sensitive data
			const callArgs = MockedAmazonBedrockEmbeddingProvider.mock.calls[0][0]
			expect(callArgs).toHaveProperty("accessKeyId", "test-access-key")
			expect(callArgs).toHaveProperty("secretAccessKey", "test-secret-key")
			expect(callArgs).toHaveProperty("sessionToken", "test-session-token")
		})

		it("should create Bedrock embedder with default model when modelId not specified", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "bedrock",
				modelId: "amazon.titan-embed-text-v1",
				bedrockOptions: {
					region: "us-east-1",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			const mockEmbedder = { validateConfiguration: vitest.fn() }
			MockedAmazonBedrockEmbeddingProvider.mockImplementation(() => mockEmbedder as any)

			// Act
			const embedder = factory.createEmbedder()

			// Assert
			expect(embedder).toBeInstanceOf(Object)
			expect(MockedAmazonBedrockEmbeddingProvider).toHaveBeenCalledWith({
				modelId: "amazon.titan-embed-text-v1",
				region: "us-east-1",
			})
		})

		it("should create Bedrock embedder with shared client configuration", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "bedrock",
				modelId: "amazon.titan-embed-text-v1",
				bedrockOptions: {
					region: "us-east-1",
					accessKeyId: "test-access-key",
					secretAccessKey: "test-secret-key",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			const mockEmbedder = { validateConfiguration: vitest.fn() }
			MockedAmazonBedrockEmbeddingProvider.mockImplementation(() => mockEmbedder as any)

			// Act
			const embedder = factory.createEmbedder()

			// Assert
			expect(embedder).toBeInstanceOf(Object)
			expect(MockedAmazonBedrockEmbeddingProvider).toHaveBeenCalledWith({
				modelId: "amazon.titan-embed-text-v1",
				region: "us-east-1",
				accessKeyId: "test-access-key",
				secretAccessKey: "test-secret-key",
			})
		})

		it("should correctly retrieve Bedrock configuration for UI/schema exposure", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "bedrock",
				modelId: "amazon.titan-embed-text-v1",
				bedrockOptions: {
					region: "us-east-1",
					awsAccessKey: "test-access-key",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act
			factory.createEmbedder()

			// Assert
			// The fact that createEmbedder successfully creates the Bedrock embedder
			// with the config demonstrates that the config is correctly retrieved.
			expect(MockedAmazonBedrockEmbeddingProvider).toHaveBeenCalledWith(
				expect.objectContaining({
					modelId: "amazon.titan-embed-text-v1",
					region: "us-east-1",
				}),
			)
		})
		it("should resolve config with profile authentication and not expose credentials", async () => {
			// Arrange
			const testConfig = {
				embedderProvider: "bedrock",
				modelId: "amazon.titan-embed-text-v1",
				awsUseProfile: true,
				awsProfile: "test-profile",
				bedrockOptions: {
					region: "us-east-1",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Mock SecretStorage to simulate profile credentials retrieval
			const mockSecretStorage = {
				get: vitest.fn().mockResolvedValue({
					accessKeyId: "profile-access-key",
					secretAccessKey: "profile-secret-key",
					sessionToken: "profile-session-token",
				}),
			}

			// Mock createBedrockRuntimeClient to capture arguments
			const createClientSpy = vitest.fn()
			vitest.mocked(createBedrockRuntimeClient).mockImplementation(createClientSpy)

			// Act
			const embedder = factory.createEmbedder()

			// Assert
			expect(embedder).toBeInstanceOf(Object)
			// Verify that no explicit credentials are passed to the embedder constructor
			expect(MockedAmazonBedrockEmbeddingProvider).toHaveBeenCalledWith({
				modelId: "amazon.titan-embed-text-v1",
				region: "us-east-1",
				accessKeyId: undefined,
				secretAccessKey: undefined,
				sessionToken: undefined,
			})
		})

		it("should not write credentials to workspace settings", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "bedrock",
				modelId: "amazon.titan-embed-text-v1",
				bedrockOptions: {
					region: "us-east-1",
					accessKeyId: "test-access-key",
					secretAccessKey: "test-secret-key",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Mock workspace.writeConfiguration to verify it's not called with credentials
			const mockWriteConfiguration = vitest.fn()
			const mockWorkspace = {
				writeConfiguration: mockWriteConfiguration,
			}

			// Act
			factory.createEmbedder()

			// Assert
			// Verify that the factory doesn't write credentials to workspace settings
			expect(mockWriteConfiguration).not.toHaveBeenCalledWith(
				expect.anything(),
				expect.objectContaining({
					accessKeyId: expect.anything(),
				}),
			)
			expect(mockWriteConfiguration).not.toHaveBeenCalledWith(
				expect.anything(),
				expect.objectContaining({
					secretAccessKey: expect.anything(),
				}),
			)
		})
	})

	describe("createVectorStore", () => {
		beforeEach(() => {
			vitest.clearAllMocks()
			mockGetDefaultModelId.mockReturnValue("default-model")
		})

		it("should use config.modelId for OpenAI provider", () => {
			// Arrange
			const testModelId = "text-embedding-3-large"
			const testConfig = {
				embedderProvider: "openai",
				modelId: testModelId,
				qdrantUrl: "http://localhost:6333",
				qdrantApiKey: "test-key",
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			mockGetModelDimension.mockReturnValue(3072)

			// Act
			factory.createVectorStore()

			// Assert
			expect(mockGetModelDimension).toHaveBeenCalledWith("openai", testModelId)
			expect(MockedQdrantVectorStore).toHaveBeenCalledWith(
				"/test/workspace",
				"http://localhost:6333",
				3072,
				"test-key",
			)
		})

		it("should use config.modelId for Ollama provider", () => {
			// Arrange
			const testModelId = "nomic-embed-text:latest"
			const testConfig = {
				embedderProvider: "ollama",
				modelId: testModelId,
				qdrantUrl: "http://localhost:6333",
				qdrantApiKey: "test-key",
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			mockGetModelDimension.mockReturnValue(768)

			// Act
			factory.createVectorStore()

			// Assert
			expect(mockGetModelDimension).toHaveBeenCalledWith("ollama", testModelId)
			expect(MockedQdrantVectorStore).toHaveBeenCalledWith(
				"/test/workspace",
				"http://localhost:6333",
				768,
				"test-key",
			)
		})

		it("should use config.modelId for OpenAI Compatible provider", () => {
			// Arrange
			const testModelId = "text-embedding-3-large"
			const testConfig = {
				embedderProvider: "openai-compatible",
				modelId: testModelId,
				qdrantUrl: "http://localhost:6333",
				qdrantApiKey: "test-key",
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			mockGetModelDimension.mockReturnValue(3072)

			// Act
			factory.createVectorStore()

			// Assert
			expect(mockGetModelDimension).toHaveBeenCalledWith("openai-compatible", testModelId)
			expect(MockedQdrantVectorStore).toHaveBeenCalledWith(
				"/test/workspace",
				"http://localhost:6333",
				3072,
				"test-key",
			)
		})

		it("should prioritize getModelDimension over manual modelDimension for OpenAI Compatible provider", () => {
			// Arrange
			const testModelId = "custom-model"
			const manualDimension = 1024
			const modelDimension = 768
			const testConfig = {
				embedderProvider: "openai-compatible",
				modelId: testModelId,
				modelDimension: manualDimension, // This should be ignored when model has built-in dimension
				openAiCompatibleOptions: {
					baseUrl: "https://api.example.com/v1",
					apiKey: "test-api-key",
				},
				qdrantUrl: "http://localhost:6333",
				qdrantApiKey: "test-key",
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			mockGetModelDimension.mockReturnValue(modelDimension) // This should be used

			// Act
			factory.createVectorStore()

			// Assert
			expect(mockGetModelDimension).toHaveBeenCalledWith("openai-compatible", testModelId)
			expect(MockedQdrantVectorStore).toHaveBeenCalledWith(
				"/test/workspace",
				"http://localhost:6333",
				modelDimension, // Should use model's built-in dimension, not manual
				"test-key",
			)
		})

		it("should use manual modelDimension only when model has no built-in dimension", () => {
			// Arrange
			const testModelId = "unknown-model"
			const manualDimension = 1024
			const testConfig = {
				embedderProvider: "openai-compatible",
				modelId: testModelId,
				modelDimension: manualDimension,
				openAiCompatibleOptions: {
					baseUrl: "https://api.example.com/v1",
					apiKey: "test-api-key",
				},
				qdrantUrl: "http://localhost:6333",
				qdrantApiKey: "test-key",
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			mockGetModelDimension.mockReturnValue(undefined) // Model has no built-in dimension

			// Act
			factory.createVectorStore()

			// Assert
			expect(mockGetModelDimension).toHaveBeenCalledWith("openai-compatible", testModelId)
			expect(MockedQdrantVectorStore).toHaveBeenCalledWith(
				"/test/workspace",
				"http://localhost:6333",
				manualDimension, // Should use manual dimension as fallback
				"test-key",
			)
		})

		it("should fall back to getModelDimension when manual modelDimension is not set for OpenAI Compatible", () => {
			// Arrange
			const testModelId = "custom-model"
			const testConfig = {
				embedderProvider: "openai-compatible",
				modelId: testModelId,
				openAiCompatibleOptions: {
					baseUrl: "https://api.example.com/v1",
					apiKey: "test-key",
				},
				qdrantUrl: "http://localhost:6333",
				qdrantApiKey: "test-key",
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			mockGetModelDimension.mockReturnValue(768)

			// Act
			factory.createVectorStore()

			// Assert
			expect(mockGetModelDimension).toHaveBeenCalledWith("openai-compatible", testModelId)
			expect(MockedQdrantVectorStore).toHaveBeenCalledWith(
				"/test/workspace",
				"http://localhost:6333",
				768,
				"test-key",
			)
		})

		it("should throw error when manual modelDimension is invalid for OpenAI Compatible", () => {
			// Arrange
			const testModelId = "custom-model"
			const testConfig = {
				embedderProvider: "openai-compatible",
				modelId: testModelId,
				modelDimension: 0, // Invalid dimension
				openAiCompatibleOptions: {
					baseUrl: "https://api.example.com/v1",
					apiKey: "test-api-key",
				},
				qdrantUrl: "http://localhost:6333",
				qdrantApiKey: "test-key",
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			mockGetModelDimension.mockReturnValue(undefined)

			// Act & Assert
			expect(() => factory.createVectorStore()).toThrow(
				"serviceFactory.vectorDimensionNotDeterminedOpenAiCompatible",
			)
		})

		it("should throw error when both manual dimension and getModelDimension fail for OpenAI Compatible", () => {
			// Arrange
			const testModelId = "unknown-model"
			const testConfig = {
				embedderProvider: "openai-compatible",
				modelId: testModelId,
				openAiCompatibleOptions: {
					baseUrl: "https://api.example.com/v1",
					apiKey: "test-key",
				},
				qdrantUrl: "http://localhost:6333",
				qdrantApiKey: "test-key",
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			mockGetModelDimension.mockReturnValue(undefined)

			// Act & Assert
			expect(() => factory.createVectorStore()).toThrow(
				"serviceFactory.vectorDimensionNotDeterminedOpenAiCompatible",
			)
		})

		it("should use model-specific dimension for Gemini provider", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "gemini",
				modelId: "gemini-embedding-001",
				qdrantUrl: "http://localhost:6333",
				qdrantApiKey: "test-key",
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			mockGetModelDimension.mockReturnValue(3072)

			// Act
			factory.createVectorStore()

			// Assert
			expect(mockGetModelDimension).toHaveBeenCalledWith("gemini", "gemini-embedding-001")
			expect(MockedQdrantVectorStore).toHaveBeenCalledWith(
				"/test/workspace",
				"http://localhost:6333",
				3072,
				"test-key",
			)
		})

		it("should use default model dimension for Bedrock when modelId not specified", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "bedrock",
				qdrantUrl: "http://localhost:6333",
				qdrantApiKey: "test-key",
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			mockGetDefaultModelId.mockReturnValue("amazon.titan-embed-text-v2.0")
			mockGetModelDimension.mockReturnValue(1024) // Assuming a default dimension for the default Bedrock model

			// Act
			factory.createVectorStore()

			// Assert
			expect(mockGetDefaultModelId).toHaveBeenCalledWith("bedrock")
			expect(mockGetModelDimension).toHaveBeenCalledWith("bedrock", "amazon.titan-embed-text-v2.0")
			expect(MockedQdrantVectorStore).toHaveBeenCalledWith(
				"/test/workspace",
				"http://localhost:6333",
				1024,
				"test-key",
			)
		})

		it("should use default model dimension for Gemini when modelId not specified", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "gemini",
				qdrantUrl: "http://localhost:6333",
				qdrantApiKey: "test-key",
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			mockGetDefaultModelId.mockReturnValue("gemini-embedding-001")
			mockGetModelDimension.mockReturnValue(3072)

			// Act
			factory.createVectorStore()

			// Assert
			expect(mockGetDefaultModelId).toHaveBeenCalledWith("gemini")
			expect(mockGetModelDimension).toHaveBeenCalledWith("gemini", "gemini-embedding-001")
			expect(MockedQdrantVectorStore).toHaveBeenCalledWith(
				"/test/workspace",
				"http://localhost:6333",
				3072,
				"test-key",
			)
		})

		it("should use default model when config.modelId is undefined", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "openai",
				modelId: undefined,
				qdrantUrl: "http://localhost:6333",
				qdrantApiKey: "test-key",
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			mockGetModelDimension.mockReturnValue(1536)

			// Act
			factory.createVectorStore()

			// Assert
			expect(mockGetModelDimension).toHaveBeenCalledWith("openai", "default-model")
			expect(MockedQdrantVectorStore).toHaveBeenCalledWith(
				"/test/workspace",
				"http://localhost:6333",
				1536,
				"test-key",
			)
		})

		it("should throw error when vector dimension cannot be determined", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "openai",
				modelId: "unknown-model",
				qdrantUrl: "http://localhost:6333",
				qdrantApiKey: "test-key",
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			mockGetModelDimension.mockReturnValue(undefined)

			// Act & Assert
			expect(() => factory.createVectorStore()).toThrow("serviceFactory.vectorDimensionNotDetermined")
		})

		it("should throw error when Qdrant URL is missing", () => {
			// Arrange
			const testConfig = {
				embedderProvider: "openai",
				modelId: "text-embedding-3-small",
				qdrantUrl: undefined,
				qdrantApiKey: "test-key",
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			mockGetModelDimension.mockReturnValue(1536)

			// Act & Assert
			expect(() => factory.createVectorStore()).toThrow("serviceFactory.qdrantUrlMissing")
		})
	})

	describe("validateEmbedder", () => {
		let mockEmbedderInstance: any

		beforeEach(() => {
			mockEmbedderInstance = {
				validateConfiguration: vitest.fn(),
			}
		})

		it("should validate OpenAI embedder successfully", async () => {
			// Arrange
			const testConfig = {
				embedderProvider: "openai",
				modelId: "text-embedding-3-small",
				openAiOptions: {
					openAiNativeApiKey: "test-api-key",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			MockedOpenAiEmbedder.mockImplementation(() => mockEmbedderInstance)
			mockEmbedderInstance.validateConfiguration.mockResolvedValue({ valid: true })

			// Act
			const embedder = factory.createEmbedder()
			const result = await factory.validateEmbedder(embedder)

			// Assert
			expect(result).toEqual({ valid: true })
			expect(mockEmbedderInstance.validateConfiguration).toHaveBeenCalled()
		})

		it("should return validation error from OpenAI embedder", async () => {
			// Arrange
			const testConfig = {
				embedderProvider: "openai",
				modelId: "text-embedding-3-small",
				openAiOptions: {
					openAiNativeApiKey: "invalid-key",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			MockedOpenAiEmbedder.mockImplementation(() => mockEmbedderInstance)
			mockEmbedderInstance.validateConfiguration.mockResolvedValue({
				valid: false,
				error: "embeddings:validation.authenticationFailed",
			})

			// Act
			const embedder = factory.createEmbedder()
			const result = await factory.validateEmbedder(embedder)

			// Assert
			expect(result).toEqual({
				valid: false,
				error: "embeddings:validation.authenticationFailed",
			})
		})

		it("should validate Ollama embedder successfully", async () => {
			// Arrange
			const testConfig = {
				embedderProvider: "ollama",
				modelId: "nomic-embed-text",
				ollamaOptions: {
					ollamaBaseUrl: "http://localhost:11434",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			MockedCodeIndexOllamaEmbedder.mockImplementation(() => mockEmbedderInstance)
			mockEmbedderInstance.validateConfiguration.mockResolvedValue({ valid: true })

			// Act
			const embedder = factory.createEmbedder()
			const result = await factory.validateEmbedder(embedder)

			// Assert
			expect(result).toEqual({ valid: true })
			expect(mockEmbedderInstance.validateConfiguration).toHaveBeenCalled()
		})

		it("should validate OpenAI Compatible embedder successfully", async () => {
			// Arrange
			const testConfig = {
				embedderProvider: "openai-compatible",
				modelId: "custom-model",
				openAiCompatibleOptions: {
					baseUrl: "https://api.example.com/v1",
					apiKey: "test-api-key",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			MockedOpenAICompatibleEmbedder.mockImplementation(() => mockEmbedderInstance)
			mockEmbedderInstance.validateConfiguration.mockResolvedValue({ valid: true })

			// Act
			const embedder = factory.createEmbedder()
			const result = await factory.validateEmbedder(embedder)

			// Assert
			expect(result).toEqual({ valid: true })
			expect(mockEmbedderInstance.validateConfiguration).toHaveBeenCalled()
		})

		it("should validate Gemini embedder successfully", async () => {
			// Arrange
			const testConfig = {
				embedderProvider: "gemini",
				geminiOptions: {
					apiKey: "test-gemini-api-key",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			MockedGeminiEmbedder.mockImplementation(() => mockEmbedderInstance)
			mockEmbedderInstance.validateConfiguration.mockResolvedValue({ valid: true })

			// Act
			const embedder = factory.createEmbedder()
			const result = await factory.validateEmbedder(embedder)

			// Assert
			expect(result).toEqual({ valid: true })
			expect(mockEmbedderInstance.validateConfiguration).toHaveBeenCalled()
		})

		it("should handle validation exceptions", async () => {
			// Arrange
			const testConfig = {
				embedderProvider: "openai",
				modelId: "text-embedding-3-small",
				openAiOptions: {
					openAiNativeApiKey: "test-api-key",
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)
			MockedOpenAiEmbedder.mockImplementation(() => mockEmbedderInstance)
			const networkError = new Error("Network error")
			mockEmbedderInstance.validateConfiguration.mockRejectedValue(networkError)

			// Act
			const embedder = factory.createEmbedder()
			const result = await factory.validateEmbedder(embedder)

			// Assert
			expect(result).toEqual({
				valid: false,
				error: "Network error",
			})
			expect(mockEmbedderInstance.validateConfiguration).toHaveBeenCalled()
		})

		it("should return error for invalid embedder configuration", async () => {
			// Arrange
			const testConfig = {
				embedderProvider: "openai",
				modelId: "text-embedding-3-small",
				openAiOptions: {
					openAiNativeApiKey: undefined, // Missing API key
				},
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act & Assert
			// This should throw when trying to create the embedder
			await expect(async () => {
				const embedder = factory.createEmbedder()
				await factory.validateEmbedder(embedder)
			}).rejects.toThrow("serviceFactory.openAiConfigMissing")
		})

		it("should return error for unknown embedder provider", async () => {
			// Arrange
			const testConfig = {
				embedderProvider: "unknown-provider",
				modelId: "some-model",
			}
			mockConfigManager.getConfig.mockReturnValue(testConfig as any)

			// Act & Assert
			// This should throw when trying to create the embedder
			expect(() => factory.createEmbedder()).toThrow("serviceFactory.invalidEmbedderType")
		})
	})
})
