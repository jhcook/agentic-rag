import { StatusBar } from 'expo-status-bar';
import { useState } from 'react';
import { StyleSheet, Text, View, TextInput, TouchableOpacity, ScrollView, SafeAreaView, ActivityIndicator, KeyboardAvoidingView, Platform, Linking } from 'react-native';
import { Send, Search, FileText, LogIn } from 'lucide-react-native';

// CONFIGURATION
// For Android Emulator use 'http://10.0.2.2:8001/api'
// For iOS Simulator use 'http://localhost:8001/api'
// For Physical Device use 'http://YOUR_LAN_IP:8001/api'
const API_BASE = 'http://localhost:8001/api';

export default function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState(null);
  const [sources, setSources] = useState([]);
  const [error, setError] = useState(null);

  const handleLogin = () => {
    Linking.openURL(`${API_BASE}/auth/login`);
  };

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setAnswer(null);
    setSources([]);

    try {
      const response = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query, k: 5 }),
      });

      if (!response.ok) {
        throw new Error(`HTTP Error: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.answer) {
        setAnswer(data.answer);
      } else if (data.choices && data.choices[0]?.message?.content) {
        setAnswer(data.choices[0].message.content);
      } else {
        setAnswer(JSON.stringify(data, null, 2));
      }

      if (data.sources) {
        setSources(data.sources);
      }

    } catch (err) {
      setError(err.message || 'Failed to connect to backend');
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="auto" />
      
      <View style={styles.header}>
        <View>
          <Text style={styles.title}>Agentic RAG</Text>
          <Text style={styles.subtitle}>Mobile Client</Text>
        </View>
        <TouchableOpacity onPress={handleLogin} style={styles.loginButton}>
          <LogIn size={20} color="#007AFF" />
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.content} contentContainerStyle={styles.contentContainer}>
        {error && (
          <View style={styles.errorCard}>
            <Text style={styles.errorText}>{error}</Text>
            <Text style={styles.errorHint}>Check if backend is running at {API_BASE}</Text>
          </View>
        )}

        {!answer && !loading && !error && (
          <View style={styles.placeholder}>
            <Search size={48} color="#ccc" />
            <Text style={styles.placeholderText}>Ask a question about your documents</Text>
          </View>
        )}

        {loading && (
          <View style={styles.loading}>
            <ActivityIndicator size="large" color="#007AFF" />
            <Text style={styles.loadingText}>Searching documents...</Text>
          </View>
        )}

        {answer && (
          <View style={styles.resultCard}>
            <Text style={styles.answerTitle}>Answer</Text>
            <Text style={styles.answerText}>{answer}</Text>
            
            {sources.length > 0 && (
              <View style={styles.sourcesSection}>
                <Text style={styles.sourcesTitle}>Sources</Text>
                {sources.map((source, index) => (
                  <View key={index} style={styles.sourceItem}>
                    <FileText size={14} color="#666" />
                    <Text style={styles.sourceText} numberOfLines={1}>
                      {source.split('/').pop()}
                    </Text>
                  </View>
                ))}
              </View>
            )}
          </View>
        )}
      </ScrollView>

      <KeyboardAvoidingView 
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        style={styles.inputContainer}
      >
        <View style={styles.inputWrapper}>
          <TextInput
            style={styles.input}
            placeholder="Ask a question..."
            value={query}
            onChangeText={setQuery}
            onSubmitEditing={handleSearch}
            returnKeyType="search"
          />
          <TouchableOpacity 
            style={[styles.sendButton, !query.trim() && styles.sendButtonDisabled]} 
            onPress={handleSearch}
            disabled={!query.trim() || loading}
          >
            <Send size={20} color="#fff" />
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    padding: 20,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  loginButton: {
    padding: 10,
    backgroundColor: '#f0f8ff',
    borderRadius: 8,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  subtitle: {
    fontSize: 14,
    color: '#666',
  },
  content: {
    flex: 1,
  },
  contentContainer: {
    padding: 20,
    paddingBottom: 100,
  },
  placeholder: {
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 100,
    opacity: 0.7,
  },
  placeholderText: {
    marginTop: 10,
    fontSize: 16,
    color: '#999',
  },
  loading: {
    alignItems: 'center',
    marginTop: 50,
  },
  loadingText: {
    marginTop: 10,
    color: '#666',
  },
  errorCard: {
    backgroundColor: '#ffebee',
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#ffcdd2',
  },
  errorText: {
    color: '#c62828',
    fontWeight: 'bold',
  },
  errorHint: {
    color: '#c62828',
    fontSize: 12,
    marginTop: 5,
  },
  resultCard: {
    backgroundColor: '#fff',
    padding: 20,
    borderRadius: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  answerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  answerText: {
    fontSize: 16,
    lineHeight: 24,
    color: '#444',
  },
  sourcesSection: {
    marginTop: 20,
    paddingTop: 15,
    borderTopWidth: 1,
    borderTopColor: '#eee',
  },
  sourcesTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
    marginBottom: 10,
  },
  sourceItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
    backgroundColor: '#f9f9f9',
    padding: 8,
    borderRadius: 6,
  },
  sourceText: {
    marginLeft: 8,
    fontSize: 12,
    color: '#555',
    flex: 1,
  },
  inputContainer: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
    padding: 15,
  },
  inputWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
    borderRadius: 25,
    paddingHorizontal: 15,
    paddingVertical: 8,
  },
  input: {
    flex: 1,
    fontSize: 16,
    maxHeight: 100,
    paddingVertical: 5,
  },
  sendButton: {
    backgroundColor: '#007AFF',
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: 10,
  },
  sendButtonDisabled: {
    backgroundColor: '#ccc',
  },
});
