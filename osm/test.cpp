#include <iostream>
#include <sstream>
#include <cmath>
#include <map>
#include <vector>
#include <set>
#include <queue>
#include <limits>
#include <chrono>
#include <random>
#include "tinyxml/tinyxml.h"
#include "jsoncpp/json/json.h"
#include <emscripten/bind.h> // 这个include在ide里面有可能会报错但是可以正常编译

using namespace emscripten;

using namespace std;
using namespace std::chrono;
/* 编译命令
emcc tinyxml/tinyxml.cpp tinyxml/tinystr.cpp tinyxml/tinyxmlerror.cpp tinyxml/tinyxmlparser.cpp jsoncpp/json_reader.cpp jsoncpp/json_value.cpp jsoncpp/json_writer.cpp -s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=4GB test.cpp -o frontend/test.js -lembind --preload-file mapFD
开启异常捕获：NO_DISABLE_EXCEPTION_CATCHING
*/
// 改为了类似json的格式方便数据转换
Json::Value nodes;
Json::Value ways;
Json::Value bounds;
Json::Value edges;
map<string, vector<pair<string, double>>> adjacencyList; //邻接表
Json::Value adjList;

// 数学处理

// 将度转换为弧度
double deg2rad(double deg) {
    return deg * M_PI / 180.0;
}

// Haversine 公式
double haversineDistance(double lat1, double lon1, double lat2, double lon2) {
    double R = 6371.0; // 地球半径（千米）
    double dLat = deg2rad(lat2 - lat1);
    double dLon = deg2rad(lon2 - lon1);
    lat1 = deg2rad(lat1);
    lat2 = deg2rad(lat2);

    double a = sin(dLat / 2) * sin(dLat / 2) +
               sin(dLon / 2) * sin(dLon / 2) * cos(lat1) * cos(lat2);
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));
    return R * c; // 距离（千米）
}

// 初始化，和上一版内容基本相同
void load()
{
    TiXmlDocument tinyXmlDoc("map2");
    tinyXmlDoc.LoadFile();
    TiXmlElement *root = tinyXmlDoc.RootElement();
    //读取bounds
    TiXmlElement *boundsElement = root->FirstChildElement("bounds");
    bounds["minlat"] = boundsElement->Attribute("minlat");
    bounds["minlon"] = boundsElement->Attribute("minlon");
    bounds["maxlat"] = boundsElement->Attribute("maxlat");
    bounds["maxlon"] = boundsElement->Attribute("maxlon");
    /*
    nodes的格式:可以根据node id索引node的经纬度
    {
        86687514:{id: '86687514', lat: '31.3007964', lon: '121.5067427'}
        86687515:{id: '86687515', lat: '31.2979531', lon: '121.4978776'}
        ...
    }
    */
    // 读nodes
    TiXmlElement *nodeElement = root->FirstChildElement("node");
    for (; nodeElement; nodeElement = nodeElement->NextSiblingElement("node"))
    {
        Json::Value node;
        node["id"] = nodeElement->Attribute("id");
        node["lon"] = nodeElement->Attribute("lon");
        node["lat"] = nodeElement->Attribute("lat");
        nodes[nodeElement->Attribute("id")] = node;
    }
    /*
    ways的格式：根据way id索引way
    nodes的格式同上
    {
        27203811:{id: '27203811', nodes: Array(6), tags: {…}}
        39408800:{id: '39408800', nodes: Array(27), tags: {…}}        
    }    
    tags的格式如下:
    {
        cycleway:"opposite_lane"
        cycleway:right:"lane"
        highway:"tertiary"
        lit:"yes"
        maxspeed:"40"
        name:"政通路"
        name:en:"Zhengtong Road"
        oneway:"yes"        
    }

    */
    // 读way
    TiXmlElement *wayElement = root->FirstChildElement("way");
    for (; wayElement; wayElement = wayElement->NextSiblingElement("way"))
    {
        Json::Value way;
        way["id"] = wayElement->Attribute("id");

        Json::Value wayNodes;
        TiXmlElement *childNode = wayElement->FirstChildElement("nd");
        for (; childNode; childNode = childNode->NextSiblingElement("nd"))
        {
            string ref = childNode->Attribute("ref");
            wayNodes.append(nodes[ref]);
        }
        way["nodes"] = wayNodes;

        Json::Value wayTags;
        TiXmlElement *childTag = wayElement->FirstChildElement("tag");
        for (; childTag; childTag = childTag->NextSiblingElement("tag"))
        {
            string name = childTag->Attribute("k");
            string value = childTag->Attribute("v");
            wayTags[name] = value;
        }
        way["tags"] = wayTags;

        ways[wayElement->Attribute("id")] = way;
    }

    // 建立邻接表
    // 两点之间的最短路应当是根据way当中带有highway tag的way决定
    // 一个node的邻居是什么？
    for (auto wayItr = ways.begin(); wayItr != ways.end(); ++wayItr) {
        const Json::Value& way = *wayItr;
        const Json::Value& wayNodes = way["nodes"];
        const Json::Value& wayTags = way["tags"];
        if (wayTags.isMember("highway")) {
            for (auto i = 0; i < wayNodes.size() - 1; ++i) {
                string currentNodeId = wayNodes[i]["id"].asString();
                string nextNodeId = wayNodes[i + 1]["id"].asString();
                // std::cout << currentNodeId << " " << nextNodeId << std::endl;
                double lat1 = stod(nodes[currentNodeId]["lat"].asString());
                double lon1 = stod(nodes[currentNodeId]["lon"].asString());
                double lat2 = stod(nodes[nextNodeId]["lat"].asString());
                double lon2 = stod(nodes[nextNodeId]["lon"].asString());

                double distance = haversineDistance(lat1, lon1, lat2, lon2);
                // std::cout << distance << std::endl;
                adjacencyList[currentNodeId].push_back(make_pair(nextNodeId, distance));
                // 如果不是单向道路
                if (!wayTags.isMember("oneway") || wayTags["oneway"].asString() != "yes") {
                    adjacencyList[nextNodeId].push_back(make_pair(currentNodeId, distance));
                }
            }
        }
    }

}
// 将邻接表转换成json
Json::Value adjacencyListToJson(const map<string, vector<pair<string, double>>>& adjacencyList) {
    Json::Value adjacencyListJson;

    // 遍历邻接表中的每个节点
    for (const auto& node : adjacencyList) {
        const string& nodeId = node.first;
        const auto& neighbors = node.second;

        // 为每个节点创建一个 JSON 数组来存储其邻接节点
        Json::Value neighborsJson(Json::arrayValue);
        for (const auto& neighbor : neighbors) {
            const string& neighborId = neighbor.first;
            double distance = neighbor.second;

            // 创建一个 JSON 对象来存储邻接节点的信息
            Json::Value neighborJson;
            neighborJson["id"] = neighborId;
            neighborJson["distance"] = distance;

            neighborsJson.append(neighborJson);
        }

        // 将邻接节点数组添加到邻接表 JSON 对象
        adjacencyListJson[nodeId] = neighborsJson;
    }

    return adjacencyListJson;
}

// A* 算法实现
vector<string> aStar(const string& startNode, const string& endNode) {
    // 小根堆存储{fScore,id}在logn时间内获取最小值
    priority_queue<pair<double, string>, vector<pair<double, string>>, greater<>> pq;
    map<string, double> gScore;
    map<string, double> fScore;
    map<string, string> cameFrom; // 方便回溯
    set<string> openSet; // 追踪优先队列中的节点

    // 初始化所有节点的 gScore 和 fScore 为无穷大
    for (const auto& node : adjacencyList) {
        gScore[node.first] = numeric_limits<double>::max();
        fScore[node.first] = numeric_limits<double>::max();
    }

    // 起点的得分
    gScore[startNode] = 0; // gScore代表从起点到当前点的最短路长度
    fScore[startNode] = haversineDistance(
        stod(nodes[startNode]["lat"].asString()),
        stod(nodes[startNode]["lon"].asString()),
        stod(nodes[endNode]["lat"].asString()),
        stod(nodes[endNode]["lon"].asString())
    ); // fScore代表从起点到当前点的最短路长度+从当前点到终点的直线（球面）路径长度

    pq.push(make_pair(fScore[startNode], startNode));
    openSet.insert(startNode);

    while (!pq.empty()) {
        string current = pq.top().second;
        pq.pop();
        openSet.erase(current);

        if (current == endNode) {
            // 找到终点，重建路径
            vector<string> path;
            while (current != startNode) {
                path.push_back(current);
                current = cameFrom[current];
            }
            path.push_back(startNode);
            reverse(path.begin(), path.end());
            return path;
        }

        for (const auto& neighbor : adjacencyList[current]) {
            string neighborId = neighbor.first;
            double tentative_gScore = gScore[current] + neighbor.second; // 根据已知最短路集合的点得到的邻接点的新gScore

            if (tentative_gScore < gScore[neighborId]) {
                cameFrom[neighborId] = current;
                gScore[neighborId] = tentative_gScore;
                fScore[neighborId] = tentative_gScore + haversineDistance(
                    stod(nodes[neighborId]["lat"].asString()),
                    stod(nodes[neighborId]["lon"].asString()),
                    stod(nodes[endNode]["lat"].asString()),
                    stod(nodes[endNode]["lon"].asString())
                );
                // 注意要特判neighborId不在set中
                if (openSet.find(neighborId) == openSet.end()) {
                    pq.push(make_pair(fScore[neighborId], neighborId));
                    openSet.insert(neighborId);
                }
            }
        }
    }

    return vector<string>(); // 如果没有找到路径，返回空路径
}

// 可以做成aStar函数的多态来实现，但是时间原因未来得及
vector<vector<string>> aStar_test(const string& startNode, const string& endNode) {
    priority_queue<pair<double, string>, vector<pair<double, string>>, greater<>> pq;
    map<string, double> gScore, fScore;
    map<string, vector<string>> cameFrom;
    set<string> openSet;

    // 初始化
    for (const auto& node : adjacencyList) {
        gScore[node.first] = numeric_limits<double>::max();
        fScore[node.first] = numeric_limits<double>::max();
    }
    gScore[startNode] = 0;
    fScore[startNode] = haversineDistance(
        stod(nodes[startNode]["lat"].asString()), 
        stod(nodes[startNode]["lon"].asString()), 
        stod(nodes[endNode]["lat"].asString()), 
        stod(nodes[endNode]["lon"].asString())
    );
    pq.push(make_pair(fScore[startNode], startNode));
    cameFrom[startNode] = {startNode};

    while (!pq.empty()) {
        string current = pq.top().second;
        pq.pop();

        if (current == endNode) {
            break; // 找到终点时停止搜索
        }

        for (const auto& neighbor : adjacencyList[current]) {
            string neighborId = neighbor.first;
            double tentative_gScore = gScore[current] + neighbor.second;

            if (tentative_gScore < gScore[neighborId]) {
                gScore[neighborId] = tentative_gScore;
                fScore[neighborId] = tentative_gScore + haversineDistance(
                    stod(nodes[neighborId]["lat"].asString()), 
                    stod(nodes[neighborId]["lon"].asString()), 
                    stod(nodes[endNode]["lat"].asString()), 
                    stod(nodes[endNode]["lon"].asString())
                );
                cameFrom[neighborId] = cameFrom[current];
                cameFrom[neighborId].push_back(neighborId);

                if (openSet.find(neighborId) == openSet.end()) {
                    pq.push(make_pair(fScore[neighborId], neighborId));
                    openSet.insert(neighborId);
                }
            }
        }
    }

    // 提取所有到达的节点的最短路径
    vector<vector<string>> allShortestPaths;
    for (const auto& node : cameFrom) {
        allShortestPaths.push_back(node.second);
    }

    return allShortestPaths;
}



// dijkstra算法实现
vector<string> dijkstra(const string& startNode, const string& endNode) {
    // 使用优先队列存储（到达成本，节点ID）
    priority_queue<pair<double, string>, vector<pair<double, string>>, greater<>> pq;
    // 存储到每个节点的最短距离
    map<string, double> distances;
    // 存储到每个节点的最短路径
    map<string, string> previous;
    // 初始化
    for (const auto& node : adjacencyList) {
        distances[node.first] = numeric_limits<double>::max();
    }
    distances[startNode] = 0;
    pq.push(make_pair(0, startNode));
    // Dijkstra 算法
    while (!pq.empty()) {
        string currentNode = pq.top().second;
        // std::cout << "current node " << currentNode << std::endl;
        double currentDistance = pq.top().first;
        // std::cout << "current distance " << currentDistance << std::endl;
        pq.pop();

        // 检查当前节点是否存在于邻接表中
        if (adjacencyList.find(currentNode) == adjacencyList.end()) {
            return vector<string>(); // 返回一个空的路径表示未找到路径
        }
        if (currentNode == endNode) {
            break; // 找到目标节点，退出循环
        }
        if (currentDistance > distances[currentNode]) {
            continue;
        }

        for (const auto& neighbor : adjacencyList.at(currentNode)) {
            // std::cout << "neighbour " << neighbor.first << std::endl;
            double newDistance = currentDistance + neighbor.second;
            if (newDistance < distances[neighbor.first]) {
                distances[neighbor.first] = newDistance;
                previous[neighbor.first] = currentNode;
                pq.push(make_pair(newDistance, neighbor.first));
            }
        }
    }
    // std::cout << "out of loop" << std::endl;
    // 检查是否找到了路径
    if (previous.find(endNode) == previous.end()) {
        // 没有找到路径
        return vector<string>();
    }
    // 构建最短路径
    vector<string> path;
    string current = endNode;
    while (current != startNode) {
        path.push_back(current);
        current = previous[current];
    }
    path.push_back(startNode);
    reverse(path.begin(), path.end());
    return path;
}
// dijkstra返回所有找到的最短路
vector<vector<string>> dijkstra_test(const string& startNode, const string& endNode) {
    priority_queue<pair<double, string>, vector<pair<double, string>>, greater<>> pq;
    map<string, double> distances;
    map<string, vector<vector<string>>> all_paths;

    // 初始化
    for (const auto& node : adjacencyList) {
        distances[node.first] = numeric_limits<double>::max();
    }
    distances[startNode] = 0;
    pq.push(make_pair(0, startNode));
    all_paths[startNode].push_back({startNode});

    // Dijkstra 算法
    while (!pq.empty()) {
        string currentNode = pq.top().second;
        double currentDistance = pq.top().first;
        pq.pop();

        if (adjacencyList.find(currentNode) == adjacencyList.end()) {
            continue; // 未找到路径
        }
        if (currentDistance > distances[currentNode]) {
            continue; // 不是更短的路径
        }

        for (const auto& neighbor : adjacencyList.at(currentNode)) {
            string neighborId = neighbor.first;
            double newDistance = currentDistance + neighbor.second;

            if (newDistance <= distances[neighborId]) {
                if (newDistance < distances[neighborId]) {
                    all_paths[neighborId].clear();
                    distances[neighborId] = newDistance;
                }
                // 更新所有到达当前节点的路径，以包含邻居
                for (auto path : all_paths[currentNode]) {
                    path.push_back(neighborId);
                    all_paths[neighborId].push_back(path);
                }
                pq.push(make_pair(newDistance, neighborId));
            }
        }

        if (currentNode == endNode) {
            break; // 找到目标节点，退出循环
        }
    }
    // 收集并返回所有找到的路径
    vector<vector<string>> collectedPaths;
    for (const auto& path : all_paths) {
        for (const auto& p : path.second) {
            collectedPaths.push_back(p);
        }
    }
    return collectedPaths;
}

// Yen's Algorithm to find the top K shortest paths
vector<vector<string>> yenTopK(const string& startNode, const string& endNode, int k) {
    vector<vector<string>> topKPaths;
    // 第一条最短路径
    vector<string> bestPath = aStar(startNode, endNode);
    if (bestPath.empty()) {
        return topKPaths; // 如果找不到最短路径，则返回空列表
    }
    topKPaths.push_back(bestPath);

    // 用于存储候选路径
    priority_queue<pair<double, vector<string>>, vector<pair<double, vector<string>>>, greater<>> candidates;

    for (int i = 1; i < k; ++i) {
        for (size_t j = 0; j < bestPath.size() - 1; ++j) {
            string spurNode = bestPath[j];
            vector<string> rootPath(bestPath.begin(), bestPath.begin() + j + 1);

            // 保存和删除所有到达spurNode的路径
            vector<pair<string, double>> removedEdges;
            for (auto& path : topKPaths) {
                if (path.size() > j && vector<string>(path.begin(), path.begin() + j + 1) == rootPath) {
                    string nextNode = path[j + 1];
                    for (auto it = adjacencyList[spurNode].begin(); it != adjacencyList[spurNode].end(); ++it) {
                        if (it->first == nextNode) {
                            removedEdges.push_back(*it);
                            it = adjacencyList[spurNode].erase(it);
                            break;
                        }
                    }
                }
            }

            // 计算从spurNode到endNode的最短路径
            vector<string> spurPath = aStar(spurNode, endNode);
            if (!spurPath.empty()) {
                vector<string> totalPath = rootPath;
                totalPath.insert(totalPath.end(), spurPath.begin() + 1, spurPath.end());
                double pathDistance = 0; // 这里需要计算 totalPath 的总距离
                candidates.push(make_pair(pathDistance, totalPath));
            }

            // 恢复删除的边
            for (const auto& edge : removedEdges) {
                adjacencyList[spurNode].push_back(edge);
            }
        }

        if (candidates.empty()) {
            break; // 没有更多路径
        }

        bestPath = candidates.top().second;
        candidates.pop();
        topKPaths.push_back(bestPath);
    }

    return topKPaths;
}

// 性能测试
// 生成随机节点
string getRandomNode() {
    auto it = begin(adjacencyList);
    advance(it, rand() % adjacencyList.size());
    return it->first;
}
// 比较sStar函数和dijkstra函数的运行时间
void compare_a_and_d(int numTrials) {
    double totalTimeAStar = 0.0;
    double totalTimeDijkstra = 0.0;
    srand(time(nullptr)); // 初始化随机数生成器

    for (int i = 0; i < numTrials; ++i) {
        // 随机选择起点和终点
        string startNode = getRandomNode();
        string endNode = getRandomNode();

        // 测试 aStar
        auto start = high_resolution_clock::now();
        vector<string> pathAStar = aStar(startNode, endNode);
        auto end = high_resolution_clock::now();
        totalTimeAStar += duration_cast<microseconds>(end - start).count();
        cout << "第" << i + 1 << "次测试dijkstra算法的时间是" << duration_cast<microseconds>(end - start).count() << "ms" << endl;
        // 测试 dijkstra
        start = high_resolution_clock::now();
        vector<string> pathDijkstra = dijkstra(startNode, endNode);
        end = high_resolution_clock::now();
        totalTimeDijkstra += duration_cast<microseconds>(end - start).count();
        cout << "第" << i + 1 << "次测试aStar算法的时间是" << duration_cast<microseconds>(end - start).count() << "ms" << endl;
    }

    // 计算平均时间
    double avgTimeAStar = totalTimeAStar / numTrials;
    double avgTimeDijkstra = totalTimeDijkstra / numTrials;

    // 输出结果
    cout << "Average time for aStar: " << avgTimeAStar << " microseconds" << endl;
    cout << "Average time for dijkstra: " << avgTimeDijkstra << " microseconds" << endl;
}



// vector<string> -> json
Json::Value pathToJson(const vector<string>& path) {
    Json::Value jsonPath;
    for (const string& node : path) {
        jsonPath.append(node);
    }
    return jsonPath;
}
// vector<vector<string>> -> json
Json::Value pathsToJson(const vector<vector<string>>& paths) {
    Json::Value allPathsJson(Json::arrayValue);

    // 遍历所有路径
    for (const auto& path : paths) {
        Json::Value pathJson(Json::arrayValue);

        // 将每个路径中的节点添加到 JSON 数组
        for (const auto& node : path) {
            pathJson.append(node);
        }

        // 将单个路径的 JSON 数组添加到所有路径中
        allPathsJson.append(pathJson);
    }

    return allPathsJson;
}

// 返回起点到终点的最短路
string findShortestPath(const string& startNode, const string& endNode) {
    std::cout << "in find shortest path" << std::endl;
    std::cout << startNode << " " << endNode << std::endl;
    // aStar or dijkstra
    // vector<string> path = dijkstra(startNode, endNode);
    vector<string> path = aStar(startNode, endNode);
    // std::cout << "finish dijkstra" << std::endl;
    std::cout << "find astar" << std::endl;
    Json::Value jsonpath = pathToJson(path);
    Json::StreamWriterBuilder builder;
    string s = Json::writeString(builder, jsonpath);
    return s;
}

// 返回dijkstra所有找到的最短路
string findShortestPath_dij_test(const string& startNode, const string& endNode) {
    std::cout << startNode << " " << endNode << std::endl;
    vector<vector<string>> paths = dijkstra_test(startNode, endNode);
    Json::Value jsonpaths = pathsToJson(paths);
    Json::StreamWriterBuilder builder;
    string s = Json::writeString(builder, jsonpaths);
    return s;
}
// 返回yen所找到的top-k短路
string findShortestTopK(const string& startNode, const string& endNode, int k) {
    std::cout << startNode << " " << endNode << std::endl;
    vector<vector<string>> paths = yenTopK(startNode, endNode, k);
    Json::Value jsonpaths = pathsToJson(paths);
    Json::StreamWriterBuilder builder;
    string s = Json::writeString(builder, jsonpaths);
    return s;
}
// 返回A*所有找到的最短路
string findShortestPath_aStar_test(const string& startNode, const string& endNode) {
    std::cout << startNode << " " << endNode << std::endl;
    vector<vector<string>> paths = aStar_test(startNode, endNode);
    Json::Value jsonpaths = pathsToJson(paths);
    Json::StreamWriterBuilder builder;
    string s = Json::writeString(builder, jsonpaths);
    return s;
}

//----------------------------------------不用的函数，只是为了测试交互------------------------------------------
void printAdjList()
{
    // 输出邻接表内容进行调试
    // std::cout << "hello" << std::endl;
    for (const auto& adjPair : adjacencyList) {
        const string& currentNode = adjPair.first;
        const vector<pair<string, double>>& neighbors = adjPair.second;

        cout << "Node " << currentNode << " has neighbors: " << endl;
        for (const auto& neighborPair : neighbors) {
            const string& neighborNode = neighborPair.first;
            double distance = neighborPair.second;
            cout << "  - Neighbor: " << neighborNode << " at distance: " << distance << endl;
        }
    }
}
// C++输出在html测试
void buildShortestPath(const std::string& startNode, const std::string& endNode)
{
    Json::CharReaderBuilder builder;
    Json::CharReader* reader = builder.newCharReader();
    Json::Value startNodeInfo;
    Json::Value endNodeInfo;
    std::string startNodeErrors;
    std::string endNodeErrors;
    bool parsingStartSuccessful = reader->parse(startNode.c_str(), startNode.c_str() + startNode.size(), &startNodeInfo, &startNodeErrors);
    if (!parsingStartSuccessful) {
        std::cout << "Failed to parse start JSON: " << startNodeErrors << std::endl;
        return;
    }
    bool parsingEndSuccessful = reader->parse(endNode.c_str(), endNode.c_str() + endNode.size(), &endNodeInfo, &endNodeErrors);
    if (!parsingEndSuccessful) {
        std::cout << "Failed to parse end JSON: " << endNodeErrors << std::endl;
        return;
    }
    delete reader;
    std::cout << "startNodeInfo:" << startNodeInfo["id"] << startNodeInfo["lon"] << " " << startNodeInfo["lat"] << endl;
    std::cout << "endNodeInfo:" << endNodeInfo["id"] << endNodeInfo["lon"] << " " << endNodeInfo["lat"] << endl;



}
// html传给C++，C++将反馈输出给html的测试
void processComplexData(const std::string& dataString) {
    Json::CharReaderBuilder builder;
    Json::CharReader* reader = builder.newCharReader();
    Json::Value data;
    std::string errors;

    bool parsingSuccessful = reader->parse(dataString.c_str(), dataString.c_str() + dataString.size(), &data, &errors);
    delete reader;

    if (!parsingSuccessful) {
        std::cout << "Failed to parse JSON: " << errors << std::endl;
        return;
    }

    std::cout << "Received name: " << data["name"].asString() << std::endl;
    std::cout << "Received value: " << data["value"].asInt() << std::endl;
    std::cout << "Received nested flag: " << data["nested"]["flag"].asBool() << std::endl;
    // 进一步处理数据...
}
//---------------------------------------------------------------------------------------------------------

// 返回bounds
string getBounds()
{
    Json::StreamWriterBuilder builder;
    string s = Json::writeString(builder, bounds);
    return s;
}
// 这两个函数返回的是json格式的数据
string getNodes()
{
    Json::StreamWriterBuilder builder;
    string s = Json::writeString(builder, nodes);
    return s;
}
// 返回xml中的ways
string getWays()
{
    Json::StreamWriterBuilder builder;
    string s = Json::writeString(builder, ways);
    return s;
}
// 返回邻接表
string getAdjList()
{
    Json::StreamWriterBuilder builder;
    Json::Value adjListJson = adjacencyListToJson(adjacencyList);
    string s = Json::writeString(builder, adjListJson);
    return s;
}

// emscripten对于c++代码的处理，为了让前端能调用这些函数
EMSCRIPTEN_BINDINGS()
{
    emscripten::function("load", &load);
    emscripten::function("getNodes", &getNodes);
    emscripten::function("getWays", &getWays);
    emscripten::function("getBounds", &getBounds);
    emscripten::function("processComplexData", &processComplexData);
    emscripten::function("buildShortestPath", &buildShortestPath);
    emscripten::function("printAdjList", &printAdjList);
    emscripten::function("findShortestPath", &findShortestPath);
    emscripten::function("getAdjList", &getAdjList);
    emscripten::function("findShortestPath_dij_test", &findShortestPath_dij_test);
    emscripten::function("findShortestPath_aStar_test", &findShortestPath_aStar_test);
    emscripten::function("findShortestTopK", &findShortestTopK);
    emscripten::function("compare_a_and_d", &compare_a_and_d);
}