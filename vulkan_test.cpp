#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


#include <limits>
#include <algorithm>
#include <string.h>

#include <fstream>
#include <iostream>

#include <vector>
#include <set>
#include <array>


const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const bool enableValidationLayers = true;


const int MAX_FRAMES_IN_FLIGHT = 2;


int currentFrame = 0;
int frame = 0;

const int numSpheres = 16;


bool framebufferResized = false;


GLFWwindow* window;
VkSurfaceKHR surface;
VkInstance instance;

VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
VkDevice logicalDevice;
VkQueue graphicsQueue;
VkQueue presentQueue;
VkQueue computeQueue;


VkSwapchainKHR swapChain;

std::vector<VkImage> swapChainImages;
std::vector<VkImageView> swapChainImageViews;
VkFormat swapChainImageFormat;
VkExtent2D swapChainExtent;

VkRenderPass renderPass;
VkDescriptorSetLayout descriptorSetLayout;
VkPipelineLayout pipelineLayout;
VkPipeline graphicsPipeline;

std::vector<VkFramebuffer> swapChainFramebuffers;

VkCommandPool commandPool;
std::vector<VkCommandBuffer> commandBuffers;

VkBuffer vertexBuffer;
VkDeviceMemory vertexBufferMemory;
VkBuffer indexBuffer;
VkDeviceMemory indexBufferMemory;

VkImage depthImage;
VkDeviceMemory depthImageMemory;
VkImageView depthImageView;

std::vector<VkBuffer> uniformBuffers;
std::vector<VkDeviceMemory> uniformBuffersMemory;
std::vector<void*> uniformBuffersMapped;

VkDescriptorPool descriptorPool;
std::vector<VkDescriptorSet> descriptorSets;


std::vector<VkSemaphore> imageAvailableSemaphores;
std::vector<VkSemaphore> renderFinishedSemaphores;
std::vector<VkFence> inFlightFences;

std::vector<VkFence> computeInFlightFences;
std::vector<VkSemaphore> computeFinishedSemaphores;


VkSampler imageSampler;

VkImage computeOutImage;
VkImageView computeOutImageView;
VkDeviceMemory computeOutImageMemory;

VkImage computeLastOutImage;
VkImageView computeLastOutImageView;
VkDeviceMemory computeLastOutImageMemory;

VkPipeline computePipeline;
VkPipelineLayout computePipelineLayout;
VkDescriptorSetLayout computeDescriptorSetLayout;

std::vector<VkCommandBuffer> computeCommandBuffers;

std::vector<VkDescriptorSet> computeDescriptorSets;


struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

struct vecTwo {
    alignas(8) glm::vec2 xy;
};

struct ivecOne {
    alignas(4) glm::ivec1 x;
};

struct sphere {
    alignas(16) glm::vec4 dim;
};
struct spheres {
    alignas(16) glm::vec4 dims[16];
    alignas(16) glm::ivec4 mats[16/4];
};


struct computeState{
    alignas(8) glm::vec2 screenExtent;
    alignas(4) glm::ivec1 x;
    alignas(4) glm::ivec1 numSpheres;
};

struct Vertex{
    glm::vec3 pos;
    glm::vec3 col;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};

    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;// can be VK_VERTEX_INPUT_RATE_INSTANCED
        return bindingDescription;
    }


    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, col);
        return attributeDescriptions;
    }
};

const float size = 4.0;

const std::vector<Vertex> vertices = {
    {{-size, -size, size}, {0.0f, 0.0f, 0.0f}},
    {{size, -size, size}, {0.0f, 0.0f, 1.0f}},
    {{size, size, size}, {0.0f, 1.0f, 0.0f}},
    {{-size, size, size}, {0.0f, 1.0f, 1.0f}},
    {{-size, -size, -size}, {1.0f, 0.0f, 0.0f}},
    {{size, -size, -size}, {1.0f, 0.0f, 1.0f}},
    {{size, size, -size}, {1.0f, 1.0f, 0.0f}},
    {{-size, size, -size}, {1.0f, 1.0f, 1.0f}}
};
const std::vector<uint16_t> indices = {
    //0, 1, 2, 2, 3, 0
    0,3,1,
    4,5,7,// open cube
    0,1,4,
    1,2,5,
    3,7,2,
    4,7,0,
    3,2,1,
    5,6,7,
    1,5,4,
    2,6,5,
    7,6,2,
    7,3,0
};


const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};


struct QueueStruct {
    int graphicsFamily = -1;
    int presentFamily = -1;
    int graphicsAndComputeFamily = -1;

    bool isComplete() {
        return (graphicsFamily != -1)&&(presentFamily != -1);
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

bool checkValidationLayerSupport();


void initWindow();
void initVulkan();
    void createInstance();
    void createSurface();
    void pickHardWareDevices();
        bool validDevice(VkPhysicalDevice device);
        QueueStruct findQueues(VkPhysicalDevice device);
        bool hasNeededExtensions(VkPhysicalDevice device);
        SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
    void createLogicalDevice();
    void createSwapChain();
        VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
        VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
        VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    void createImageViews();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
        static std::vector<char> readFile(const std::string& filename);
        VkShaderModule createShaderModule(const std::vector<char>& code);
    void createRenderPass();
    void createFramebuffers();
    void createCommandPool();
    void createDepthResources();
        VkFormat findDepthFormat();
    void createVertexBuffer();
        uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void createIndexBuffer();
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();
    void createCommandBuffers();
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
    void createSyncObjects();
    void createSampler();
void mainLoop();
    void updateUniformBuffer(uint32_t currentImage);
    void drawFrame();
void cleanUp();

void createComputeCommandBuffers();
void createComputeImages();
void createComputePipeLine();
void createComputeDescriptorSetLayout();
void createComputeDescriptorSets();

void recordComputeCommandBuffer(VkCommandBuffer commandBuffer,int i);

bool hasStencilComponent(VkFormat format);

void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout startLayout, VkImageLayout endLayout);

VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);

VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);

void recreateSwapChain();// incase current one is invalidated
void cleanupSwapChain();

static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,VkMemoryPropertyFlags properties, VkBuffer &returnBuffer, VkDeviceMemory &returnBufferMemory);
void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

class Model{
public:
    Model(std::vector<Vertex> vertices, std::vector<uint16_t> indices);
    ~Model();
    // only use when recording the command buffer
    void showSelf(VkCommandBuffer commandBuffer);
private:
    void setUpVertexBuffer(std::vector<Vertex> verts);
    void setUpIndexBuffer(std::vector<uint16_t> inds);

    VkBuffer mVertexBuffer;
    VkDeviceMemory mVertexBufferMemory;

    VkBuffer mIndexBuffer;
    VkDeviceMemory mIndexBufferMemory;

    int numIndicies;
    int numVerticies;

};

std::vector<Model> models;


int main() {
    initWindow();

    initVulkan();
    
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

    std::cout << extensionCount << " extensions supported\n";

    const std::vector<Vertex> verts = {
        {{-2.0f, -2.0f, -2.0f}, {1.0f, 0.0f, 0.0f}},
        {{-2.0f, -2.0f, 2.0f}, {1.0f, 0.0f, 1.0f}},
        {{-2.0f, 2.0f, -2.0f}, {0.0f, 1.0f, 0.0f}},
        {{2.0f, -2.0f, -2.0f}, {1.0f, 0.0f, 0.0f}}
    };
    const std::vector<uint16_t> inds = {
        0,1,2,
        3,1,0,
        3,2,0,
        3,2,1
    };

    //Model newModel = Model(verts,inds);
    //models.push_back(newModel);

    mainLoop();

    cleanUp();

    return 0;
}

void initWindow(){
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    //glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // due to being able to recreate the swap chain is not needed


    window = glfwCreateWindow(800, 608, "Vulkan window", nullptr, nullptr);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

void initVulkan(){
    createInstance();
    createSurface();

    pickHardWareDevices();

    createLogicalDevice();

    createSwapChain();
    createImageViews();


    createCommandPool();

    createComputeImages();// must be before descriptors made
    transitionImageLayout(computeOutImage, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    transitionImageLayout(computeLastOutImage, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);


    createSampler();

    createRenderPass();
    createDescriptorSetLayout();
    createGraphicsPipeline();


    createDepthResources();

    createFramebuffers();

    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();


    createCommandBuffers();



    createComputeDescriptorSetLayout();
    createComputeDescriptorSets();
    createComputeCommandBuffers();
    createComputePipeLine();

    createSyncObjects();
}

void createInstance(){
    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }


    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;

    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    createInfo.enabledExtensionCount = glfwExtensionCount;
    createInfo.ppEnabledExtensionNames = glfwExtensions;


    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("failed to create instance!");
    }
}

bool checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers) {
        bool layerFound = false;

        for (const VkLayerProperties& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            return false;
        }
    }

    return true;
}

void createSurface() {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
}

void pickHardWareDevices(){
    physicalDevice = VK_NULL_HANDLE;

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0){
        std::cout << "no gpus with vulkan support\n";
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }
    std::cout << deviceCount << " gpus\n";

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const VkPhysicalDevice& device : devices) {
        std::cout << device << "hi\n";
        std::cout << devices.size() << "\n";
        if (device == VK_NULL_HANDLE){
            std::cout << "??\n";
        }
        if (validDevice(device)) {
            physicalDevice = device;
            break;
        }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
}

bool validDevice(VkPhysicalDevice device){



    // version supported and other stuff
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);

    // what optional support is there
    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

    QueueStruct indcies = findQueues(device);
    if (!indcies.isComplete()){
        std::cout << "error missing vital queue\n";
        return false;
    }
    std::cout << indcies.graphicsFamily << " graphic queue\n";

    if (!hasNeededExtensions(device)){
        return false;
    }

    bool swapChainAdequate = false;
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
    swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    std::cout << swapChainSupport.formats.size() << " " << swapChainSupport.presentModes.size() << "\n";
    if (!swapChainAdequate){
        std::cout << "not swap chain adequate\n";
        return false;
    }

    return true;
}

QueueStruct findQueues(VkPhysicalDevice device){
    QueueStruct indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (const VkQueueFamilyProperties& queueFamily : queueFamilies){
        std::cout << i << " queue\n";
        std::cout << queueFamily.queueFlags << "\n";
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT){
            indices.graphicsFamily = i;
        }
        VkBool32 canPresent = false;
        std::cout << surface << "\n";
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &canPresent);

        if (canPresent){
            indices.presentFamily = i;
        }

        if((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)){
            indices.graphicsAndComputeFamily = i;
        }


        if (indices.isComplete()){
            //break;
        }

        i++;
    }

    return indices;
}

bool hasNeededExtensions(VkPhysicalDevice device){


    

    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    for (const VkExtensionProperties& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
}

void createLogicalDevice(){
    QueueStruct indices = findQueues(physicalDevice);


    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies;
    uniqueQueueFamilies.insert(indices.graphicsFamily);
    uniqueQueueFamilies.insert(indices.presentFamily);
    uniqueQueueFamilies.insert(indices.graphicsAndComputeFamily);

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures{};

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();

    createInfo.pEnabledFeatures = &deviceFeatures;


    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();


    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }



    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &logicalDevice) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!");
    }

    vkGetDeviceQueue(logicalDevice, indices.graphicsFamily, 0, &graphicsQueue);
    vkGetDeviceQueue(logicalDevice, indices.presentFamily, 0, &presentQueue);
    vkGetDeviceQueue(logicalDevice, indices.graphicsAndComputeFamily, 0, &computeQueue);


}

void createSwapChain(){
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);


    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;// +1 so dont have to wait for driver to finish as much
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {// 0 means no max
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;

    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueStruct indices = findQueues(physicalDevice);
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily, indices.presentFamily};

    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0; // Optional
        createInfo.pQueueFamilyIndices = nullptr; // Optional
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;

    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(logicalDevice, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
        throw std::runtime_error("failed to create swap chain!");
    }


    vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, swapChainImages.data());


    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
}

VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    for (const VkSurfaceFormatKHR& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    for (const VkPresentModeKHR& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return availablePresentMode;
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    } else {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        VkExtent2D actualExtent = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };

        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actualExtent;
    }
}

void createImageViews(){
    swapChainImageViews.resize(swapChainImages.size());
    for (int i = 0; i < swapChainImages.size(); i++) {
        swapChainImageViews[i] = createImageView(swapChainImages[i],swapChainImageFormat,VK_IMAGE_ASPECT_COLOR_BIT);
    }


}

VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags){
    VkImageViewCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = image;
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format = format;

    createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

    createInfo.subresourceRange.aspectMask = aspectFlags;
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = 1;
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    if (vkCreateImageView(logicalDevice, &createInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image views!");
    }

    return imageView;
}

void createDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding,3> uboLayoutBindings{};
    uboLayoutBindings[0].binding = 0;
    uboLayoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBindings[0].descriptorCount = 1;
    uboLayoutBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uboLayoutBindings[0].pImmutableSamplers = nullptr; // Optional

    uboLayoutBindings[1].binding = 1;
    uboLayoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    uboLayoutBindings[1].descriptorCount = 1;
    uboLayoutBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    uboLayoutBindings[1].pImmutableSamplers = nullptr; // Optional??

    uboLayoutBindings[2].binding = 2;
    uboLayoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBindings[2].descriptorCount = 1;
    uboLayoutBindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    uboLayoutBindings[2].pImmutableSamplers = nullptr; // Optional

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = uboLayoutBindings.size();
    layoutInfo.pBindings = uboLayoutBindings.data();

    if (vkCreateDescriptorSetLayout(logicalDevice, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
}

void createGraphicsPipeline(){
    std::vector<char> vertShaderCode = readFile("vulkan_shaders/shader_vert.spv");
    std::vector<char> fragShaderCode = readFile("vulkan_shaders/shader_frag.spv");

    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";


    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};


    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;


    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    /*VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;*/

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float) swapChainExtent.width;
    viewport.height = (float) swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f; // Optional
    depthStencil.maxDepthBounds = 1.0f; // Optional
    depthStencil.stencilTestEnable = VK_FALSE;
    depthStencil.front = {}; // Optional
    depthStencil.back = {}; // Optional

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f; // cant be more than 1 without cahnging something
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f; // Optional
    rasterizer.depthBiasClamp = 0.0f; // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f; // Optional
    multisampling.pSampleMask = nullptr; // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling.alphaToOneEnable = VK_FALSE; // Optional

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f; // Optional
    colorBlending.blendConstants[1] = 0.0f; // Optional
    colorBlending.blendConstants[2] = 0.0f; // Optional
    colorBlending.blendConstants[3] = 0.0f; // Optional


    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1; // Optional
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout; // Optional
    pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
    pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

    if (vkCreatePipelineLayout(logicalDevice, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;

    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = nullptr; // Optional
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.pDepthStencilState = &depthStencil;

    pipelineInfo.layout = pipelineLayout;

    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;

    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
    pipelineInfo.basePipelineIndex = -1; // Optional


    if (vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(logicalDevice, fragShaderModule, nullptr);
    vkDestroyShaderModule(logicalDevice, vertShaderModule, nullptr);
}


static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);//start reading from end so know how big it is

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

VkShaderModule createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(logicalDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }
    return shaderModule;
}

void createRenderPass(){
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = findDepthFormat();
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;



    if (vkCreateRenderPass(logicalDevice, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }
}


void createFramebuffers(){
    swapChainFramebuffers.resize(swapChainImageViews.size());
    for (int i = 0; i < swapChainImageViews.size(); i++){
        std::array<VkImageView,2> attachments = {
            swapChainImageViews[i],
            depthImageView
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(logicalDevice, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

void createCommandPool(){
    QueueStruct queueFamilyIndices = findQueues(physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;

    if (vkCreateCommandPool(logicalDevice, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }
}

void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(logicalDevice, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(logicalDevice, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(logicalDevice, image, imageMemory, 0);
}

VkCommandBuffer beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(logicalDevice, commandPool, 1, &commandBuffer);
}

void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL){
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL){
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_GENERAL){
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    else{
        throw std::invalid_argument("unsupported layout transition! (image)");
    }

    vkCmdPipelineBarrier(
        commandBuffer,
        sourceStage, destinationStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );


    endSingleTimeCommands(commandBuffer);
}

void createDepthResources(){
    VkFormat depthFormat = findDepthFormat();
    createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
    depthImageView = createImageView(depthImage, depthFormat,VK_IMAGE_ASPECT_DEPTH_BIT);

}

VkFormat findDepthFormat() {
    return findSupportedFormat(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
    );
}

void createSampler(){
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physicalDevice, &properties);

    samplerInfo.anisotropyEnable = VK_FALSE;//VK_TRUE;
    samplerInfo.maxAnisotropy = 1.0f;//properties.limits.maxSamplerAnisotropy;

    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;

    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;

    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;


    if (vkCreateSampler(logicalDevice, &samplerInfo, nullptr, &imageSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }

}


void createComputeCommandBuffers(){
    computeCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t) computeCommandBuffers.size();

    if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, computeCommandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
    }
}

void createComputeImages(){
    createImage(swapChainExtent.width,swapChainExtent.height,VK_FORMAT_R16G16B16A16_UNORM,VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                , VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, computeOutImage, computeOutImageMemory);
    computeOutImageView = createImageView(computeOutImage, VK_FORMAT_R16G16B16A16_UNORM,VK_IMAGE_ASPECT_COLOR_BIT);

    createImage(swapChainExtent.width,swapChainExtent.height,VK_FORMAT_R16G16B16A16_UNORM,VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                , VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, computeLastOutImage, computeLastOutImageMemory);
    computeLastOutImageView = createImageView(computeLastOutImage, VK_FORMAT_R16G16B16A16_UNORM,VK_IMAGE_ASPECT_COLOR_BIT);
}

void createComputeDescriptorSetLayout(){
    std::array<VkDescriptorSetLayoutBinding,4> layouts;
    layouts[0].binding = 0;
    layouts[0].descriptorCount = 1;
    layouts[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    layouts[0].pImmutableSamplers = nullptr;
    layouts[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    layouts[1].binding = 1;
    layouts[1].descriptorCount = 1;
    layouts[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    layouts[1].pImmutableSamplers = nullptr;
    layouts[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    layouts[2].binding = 2;
    layouts[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layouts[2].descriptorCount = 1;
    layouts[2].pImmutableSamplers = nullptr;
    layouts[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    layouts[3].binding = 3;
    layouts[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layouts[3].descriptorCount = 1;
    layouts[3].pImmutableSamplers = nullptr;
    layouts[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(layouts.size());
    layoutInfo.pBindings = layouts.data();

    if (vkCreateDescriptorSetLayout(logicalDevice, &layoutInfo, nullptr, &computeDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create compute descriptor set layout!");
    }
}

void createComputeDescriptorSets(){
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, computeDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    allocInfo.pSetLayouts = layouts.data();

    computeDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(logicalDevice, &allocInfo, computeDescriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate compute descriptor sets!");
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {


        VkDescriptorImageInfo imageInfo{};
        imageInfo.sampler = imageSampler;
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;//?? validator said this is only allowed format
        imageInfo.imageView = computeOutImageView;

        VkDescriptorImageInfo imageInfo2{};
        imageInfo2.sampler = imageSampler;
        imageInfo2.imageLayout = VK_IMAGE_LAYOUT_GENERAL;//?? validator said this is only allowed format
        imageInfo2.imageView = computeLastOutImageView;

        VkDescriptorBufferInfo bufferInfoComputeState{};
        bufferInfoComputeState.buffer = uniformBuffers[MAX_FRAMES_IN_FLIGHT*2 +i];
        bufferInfoComputeState.offset = 0;
        bufferInfoComputeState.range = sizeof(computeState);


        VkDescriptorBufferInfo bufferInfoSpheres{};
        bufferInfoSpheres.buffer = uniformBuffers[MAX_FRAMES_IN_FLIGHT*3 +i];
        bufferInfoSpheres.offset = 0;
        bufferInfoSpheres.range = sizeof(spheres);


        std::array<VkWriteDescriptorSet, 4> descriptorWrites{};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = computeDescriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pImageInfo = &imageInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = computeDescriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &imageInfo2;

        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = computeDescriptorSets[i];
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].dstArrayElement = 0;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].pBufferInfo = &bufferInfoComputeState;

        descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[3].dstSet = computeDescriptorSets[i];
        descriptorWrites[3].dstBinding = 3;
        descriptorWrites[3].dstArrayElement = 0;
        descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[3].descriptorCount = 1;
        descriptorWrites[3].pBufferInfo = &bufferInfoSpheres;

        vkUpdateDescriptorSets(logicalDevice, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void createComputePipeLine(){
    std::vector<char> computeShaderCode = readFile("vulkan_shaders/compute_ray_tracer.spv");
    VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

    VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
    computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeShaderStageInfo.module = computeShaderModule;
    computeShaderStageInfo.pName = "main";

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &computeDescriptorSetLayout;
    

    if (vkCreatePipelineLayout(logicalDevice, &pipelineLayoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create compute pipeline layout!");
    }
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = computePipelineLayout;
    pipelineInfo.stage = computeShaderStageInfo;

    if (vkCreateComputePipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create compute pipeline!");
    }


    vkDestroyShaderModule(logicalDevice, computeShaderModule, nullptr);
}

void recordComputeCommandBuffer(VkCommandBuffer commandBuffer,int i){
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[i], 0, 0);

    vkCmdDispatch(commandBuffer, swapChainExtent.width/16, swapChainExtent.height/16, 1);


    // copy result to another image so can be used later
    VkExtent3D extent{};
    extent.depth = 1;
    extent.width = swapChainExtent.width;
    extent.height = swapChainExtent.height;

    VkImageSubresourceLayers dstsr{};
    VkImageSubresourceLayers srcsr{};

    dstsr.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    dstsr.mipLevel = 0;
    dstsr.baseArrayLayer = 0;
    dstsr.layerCount = 1;

    srcsr.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    srcsr.mipLevel = 0;
    srcsr.baseArrayLayer = 0;
    srcsr.layerCount = 1;

    VkImageCopy regions{};
    regions.extent = extent;
    regions.dstSubresource = dstsr;
    regions.srcSubresource = srcsr;

    vkCmdCopyImage(commandBuffer, computeOutImage, VK_IMAGE_LAYOUT_GENERAL, computeLastOutImage, VK_IMAGE_LAYOUT_GENERAL, 1, &regions);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }
}

bool hasStencilComponent(VkFormat format)
{
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
    for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
            return format;
        } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }


    throw std::runtime_error("failed to find supported format!");
}

void createVertexBuffer(){
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                ,stagingBuffer,stagingBufferMemory);
    
    void* data;
    vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), (size_t) bufferSize);
    vkUnmapMemory(logicalDevice, stagingBufferMemory);

    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                ,vertexBuffer,vertexBufferMemory);
    

    copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

    vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
}

uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void createIndexBuffer(){
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                ,stagingBuffer,stagingBufferMemory);
    
    void* data;
    vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, indices.data(), (size_t) bufferSize);
    vkUnmapMemory(logicalDevice, stagingBufferMemory);

    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                ,indexBuffer,indexBufferMemory);
    

    copyBuffer(stagingBuffer, indexBuffer, bufferSize);

    vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
}

void createUniformBuffers(){
    VkDeviceSize bufferSize;

    const std::vector<int> numUbos = {1,1,1,1};
    const std::vector<VkDeviceSize> sizeUbos = {sizeof(UniformBufferObject),
                                                sizeof(vecTwo),
                                                sizeof(computeState),
                                                sizeof(spheres)};

    std::vector<int> cumulativeSum = {0};
    for (int i = 0; i < numUbos.size(); i++){
        cumulativeSum.push_back(numUbos[i]*MAX_FRAMES_IN_FLIGHT +cumulativeSum[i]);
        std::cout << cumulativeSum[cumulativeSum.size()-1] << "\n";
    }

    uniformBuffers.resize(cumulativeSum[cumulativeSum.size()-1]);
    uniformBuffersMemory.resize(cumulativeSum[cumulativeSum.size()-1]);
    uniformBuffersMapped.resize(cumulativeSum[cumulativeSum.size()-1]);

    int index = 0;

    for (int i = 0; i < cumulativeSum[cumulativeSum.size()-1]; i++){
        while (cumulativeSum[index] <= i){
            index++;
            bufferSize = sizeUbos[index-1];
        }
        std::cout << index << " " << i << " " << bufferSize << "\n";
        createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);

        vkMapMemory(logicalDevice, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
    }
}

void createDescriptorPool(){
    std::array<VkDescriptorPoolSize,3> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT *4);
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[2].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT *2);

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT *7);

    if (vkCreateDescriptorPool(logicalDevice, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }

}

void createDescriptorSets(){
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(logicalDevice, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        VkDescriptorBufferInfo bufferInfov2{};
        bufferInfov2.buffer = uniformBuffers[MAX_FRAMES_IN_FLIGHT +i];
        bufferInfov2.offset = 0;
        bufferInfov2.range = sizeof(vecTwo);



        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;//VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;// beacause i am accessing it as a storage image
        imageInfo.imageView = computeOutImageView;
        imageInfo.sampler = imageSampler;


        std::array<VkWriteDescriptorSet,3> descriptorWrites{};
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = descriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;

        descriptorWrites[0].pBufferInfo = &bufferInfo;
        descriptorWrites[0].pImageInfo = nullptr; // Optional
        descriptorWrites[0].pTexelBufferView = nullptr; // Optional


        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = descriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;

        descriptorWrites[1].pImageInfo = &imageInfo;
        descriptorWrites[1].pBufferInfo = nullptr; // Optional
        descriptorWrites[1].pTexelBufferView = nullptr; // Optional


        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = descriptorSets[i];
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].dstArrayElement = 0;
        
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[2].descriptorCount = 1;

        descriptorWrites[2].pBufferInfo = &bufferInfov2;
        descriptorWrites[2].pImageInfo = nullptr; // Optional
        descriptorWrites[2].pTexelBufferView = nullptr; // Optional

        vkUpdateDescriptorSets(logicalDevice, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void createCommandBuffers(){
    commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);


    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();

    if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
    }
}

void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex){
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0; // Optional
    beginInfo.pInheritanceInfo = nullptr; // Optional

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];

    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = swapChainExtent;

    std::array<VkClearValue,2> clearValues;
    clearValues[0] = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[1] = {{1.0f, 0.0f}};
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);


    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swapChainExtent.width);
    viewport.height = static_cast<float>(swapChainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);


    VkBuffer vertexBuffers[] = {vertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

    //vkCmdDraw(commandBuffer, static_cast<uint32_t>(vertices.size()), 1, 0, 0);
    vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
    //vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 1, 0, 0);

    for (int i = 0; i < models.size(); i++){
        //models[i].showSelf(commandBuffer);
    }


    vkCmdEndRenderPass(commandBuffer);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }

}


void createSyncObjects(){
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    computeInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    computeFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);



    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++){
        if (vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(logicalDevice, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS ||
            vkCreateFence(logicalDevice, &fenceInfo, nullptr, &computeInFlightFences[i]) != VK_SUCCESS ||
            vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &computeFinishedSemaphores[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create semaphores!");
        }
    }
}

void mainLoop(){
    int i = 0;
    while(!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        updateUniformBuffer(currentFrame);
        drawFrame();
        if (i%100 == 0){
            std::cout << i << "hi\n";
        }
        i++;
        //throw std::runtime_error("check");
    }


    vkDeviceWaitIdle(logicalDevice);
}

void updateUniformBuffer(uint32_t currentImage){
    UniformBufferObject ubo{};
    ubo.model = glm::rotate(glm::mat4(1.0f),frame*0.001f,glm::vec3(0.0f,0.0f,1.0f));
    ubo.view = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.proj = glm::perspective((float)M_PI/4.0f, swapChainExtent.width/(float) swapChainExtent.height, 0.1f,10.0f);
    ubo.proj[1][1] *= -1;//because it was designed for opengl y flip

    memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));

    vecTwo v2{};
    v2.xy = glm::vec2(swapChainExtent.width,swapChainExtent.height);

    memcpy(uniformBuffersMapped[MAX_FRAMES_IN_FLIGHT +currentImage], &v2, sizeof(v2));

    computeState state;
    float fov = 1;
    state.screenExtent = glm::vec2(fov,fov/swapChainExtent.width*swapChainExtent.height);
    state.x = glm::ivec1(frame);
    state.numSpheres = glm::ivec1(numSpheres);
    memcpy(uniformBuffersMapped[MAX_FRAMES_IN_FLIGHT*2 +currentImage], &state, sizeof(state));

    spheres s;
    for (int i = 6; i < numSpheres; i++){
        s.dims[i] = glm::vec4(i*0.1,0,1,0.1);
    }

    s.dims[0] = glm::vec4(0,0,4,0.5);
    s.dims[1] = glm::vec4(1,2,5,2.5);
    s.dims[2] = glm::vec4(8,-6,20,2.5);
    s.dims[3] = glm::vec4(-8,-8,16,5.5);
    s.dims[4] = glm::vec4(0,0,80,45);
    s.dims[5] = glm::vec4(0,0,2,0.1);

    for (int i = 0; i < numSpheres/4; i++){
        s.mats[i] = glm::ivec4(4*(i%2),4*(i%2)+1,4*(i%2)+2,4*(i%2)+3);
    }
    
    memcpy(uniformBuffersMapped[MAX_FRAMES_IN_FLIGHT*3 +currentImage], &s, sizeof(s));
}

void drawFrame(){
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    // compute stuff
    vkWaitForFences(logicalDevice, 1, &computeInFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    vkResetFences(logicalDevice, 1, &computeInFlightFences[currentFrame]);

    vkResetCommandBuffer(computeCommandBuffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);
    recordComputeCommandBuffer(computeCommandBuffers[currentFrame], currentFrame);

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &computeCommandBuffers[currentFrame];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &computeFinishedSemaphores[currentFrame];

    if (vkQueueSubmit(computeQueue, 1, &submitInfo, computeInFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit compute command buffer!");
    };

    // graphic stuff
    vkWaitForFences(logicalDevice, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;

    VkResult result = vkAcquireNextImageKHR(logicalDevice, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || framebufferResized) {
        framebufferResized = false;
        recreateSwapChain();
        return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }


    vkResetFences(logicalDevice, 1, &inFlightFences[currentFrame]);// here as may exit early

    vkResetCommandBuffer(commandBuffers[currentFrame], 0);
    recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

    VkSemaphore waitSemaphores[] = {computeFinishedSemaphores[currentFrame], imageAvailableSemaphores[currentFrame]};
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 2;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

    VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapChains[] = {swapChain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr; // Optional

    result = vkQueuePresentKHR(presentQueue, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
        framebufferResized = false;
        recreateSwapChain();
    } else if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to present swap chain image!");
    }


    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    frame++;
}

void cleanUp(){
    cleanupSwapChain();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(logicalDevice, renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(logicalDevice, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(logicalDevice, inFlightFences[i], nullptr);
        vkDestroyFence(logicalDevice, computeInFlightFences[i], nullptr);
        vkDestroySemaphore(logicalDevice, computeFinishedSemaphores[i], nullptr);
    }

    for (int i = 0; i < models.size(); i++){
        models[i].~Model();
    }

    vkDestroyBuffer(logicalDevice, vertexBuffer, nullptr);
    vkFreeMemory(logicalDevice, vertexBufferMemory, nullptr);
    vkDestroyBuffer(logicalDevice, indexBuffer, nullptr);
    vkFreeMemory(logicalDevice, indexBufferMemory, nullptr);

    vkDestroyCommandPool(logicalDevice, commandPool, nullptr);

    vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);

    vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);

    for (int i = 0; i < uniformBuffers.size(); i++){
        vkDestroyBuffer(logicalDevice, uniformBuffers[i], nullptr);
        vkFreeMemory(logicalDevice, uniformBuffersMemory[i], nullptr);
    }


    vkDestroyPipeline(logicalDevice, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);

    vkDestroyRenderPass(logicalDevice, renderPass, nullptr);

    vkDestroyImage(logicalDevice, depthImage, nullptr);
    vkFreeMemory(logicalDevice, depthImageMemory, nullptr);
    vkDestroyImageView(logicalDevice, depthImageView, nullptr);

    vkDestroyDescriptorSetLayout(logicalDevice, computeDescriptorSetLayout, nullptr);

    vkDestroyPipeline(logicalDevice, computePipeline, nullptr);
    vkDestroyPipelineLayout(logicalDevice, computePipelineLayout, nullptr);


    vkDestroyImage(logicalDevice, computeOutImage, nullptr);
    vkFreeMemory(logicalDevice, computeOutImageMemory, nullptr);

    vkDestroyImage(logicalDevice, computeLastOutImage, nullptr);
    vkFreeMemory(logicalDevice, computeLastOutImageMemory, nullptr);

    vkDestroySampler(logicalDevice, imageSampler, nullptr);
    vkDestroyImageView(logicalDevice, computeOutImageView, nullptr);
    vkDestroyImageView(logicalDevice, computeLastOutImageView, nullptr);
    
    vkDestroyDevice(logicalDevice, nullptr);
    
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);


    glfwDestroyWindow(window);

    glfwTerminate();
}

void recreateSwapChain() {

    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) { // wait while 0 size
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }




    vkDeviceWaitIdle(logicalDevice);

    cleanupSwapChain();

    createSwapChain();
    createImageViews();
    createFramebuffers();
}

void cleanupSwapChain(){
    for (size_t i = 0; i < swapChainFramebuffers.size(); i++) {
        vkDestroyFramebuffer(logicalDevice, swapChainFramebuffers[i], nullptr);
    }

    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        vkDestroyImageView(logicalDevice, swapChainImageViews[i], nullptr);
    }


    vkDestroySwapchainKHR(logicalDevice, swapChain, nullptr);
}

static void framebufferResizeCallback(GLFWwindow* window, int width, int height){
    framebufferResized = true;
}

void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,VkMemoryPropertyFlags properties, VkBuffer &returnBuffer, VkDeviceMemory &returnBufferMemory){
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(logicalDevice, &bufferInfo, nullptr, &returnBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create vertex buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(logicalDevice, returnBuffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &returnBufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate vertex buffer memory!");
    }


    vkBindBufferMemory(logicalDevice, returnBuffer, returnBufferMemory, 0);
}

void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0; // Optional
    copyRegion.dstOffset = 0; // Optional
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
    
    vkEndCommandBuffer(commandBuffer);
    
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(logicalDevice, commandPool, 1, &commandBuffer);
}

Model::Model(std::vector<Vertex> vertices, std::vector<uint16_t> indices){
    setUpVertexBuffer(vertices);
    setUpIndexBuffer(indices);
    numIndicies = indices.size();
    numVerticies = vertices.size();
    std::cout << numVerticies << " " << numIndicies << "\n";
}

Model::~Model(){
    vkDestroyBuffer(logicalDevice, mVertexBuffer, nullptr);
    vkFreeMemory(logicalDevice, mVertexBufferMemory, nullptr);
    vkDestroyBuffer(logicalDevice, mIndexBuffer, nullptr);
    vkFreeMemory(logicalDevice, mIndexBufferMemory, nullptr);
    
}

void Model::showSelf(VkCommandBuffer commandBuffer){
    VkBuffer vertexBuffers[] = {mVertexBuffer};
    VkDeviceSize offsets[] = {0};
    
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    
    vkCmdBindIndexBuffer(commandBuffer, mIndexBuffer, 0, VK_INDEX_TYPE_UINT16);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

    vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(numIndicies), 1, 0, 0, 0);
}

void Model::setUpVertexBuffer(std::vector<Vertex> verts){
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    VkDeviceSize bufferSize = sizeof(verts[0]) * verts.size();
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                ,stagingBuffer,stagingBufferMemory);
    
    void* data;
    vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, verts.data(), (size_t) bufferSize);
    vkUnmapMemory(logicalDevice, stagingBufferMemory);

    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                ,mVertexBuffer,mVertexBufferMemory);
    

    copyBuffer(stagingBuffer, mVertexBuffer, bufferSize);

    vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
}

void Model::setUpIndexBuffer(std::vector<uint16_t> inds){
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    VkDeviceSize bufferSize = sizeof(inds[0]) * inds.size();
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                ,stagingBuffer,stagingBufferMemory);
    
    void* data;
    vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, inds.data(), (size_t) bufferSize);
    vkUnmapMemory(logicalDevice, stagingBufferMemory);

    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                ,mIndexBuffer,mIndexBufferMemory);
    

    copyBuffer(stagingBuffer, mIndexBuffer, bufferSize);

    vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
}