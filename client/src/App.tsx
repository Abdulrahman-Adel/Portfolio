import { Switch, Route, Router } from "wouter";
import { Toaster } from "@/components/ui/toaster";
import NotFound from "@/pages/not-found";
import Home from "@/pages/Home";
import BlogIndexPage from "@/pages/BlogIndexPage";
import BlogPostPage from "@/pages/BlogPostPage";

// No custom hook needed

function AppRouter() {
  return (
    <Switch>
      <Route path="/" component={Home} />
      <Route path="/blog" component={BlogIndexPage} />
      <Route path="/blog/:slug" component={BlogPostPage} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  // Get base path from Vite's environment variables
  const base = import.meta.env.BASE_URL;

  return (
    // Use Router with the base prop
    <Router base={base}>
      <AppRouter />
      <Toaster />
    </Router>
  );
}

export default App;
