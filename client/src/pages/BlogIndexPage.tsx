import React from 'react';
import { Link } from 'react-router-dom'; // Assuming react-router-dom
// TODO: Fetch and list blog posts from the content directory

const BlogIndexPage: React.FC = () => {
  // TODO: Replace with actual post data
  const posts = [
    { slug: 'first-post', title: 'My First Blog Post' },
    { slug: 'second-post', title: 'Another Interesting Article' },
  ];

  return (
    <div>
      <h1>Blog / Articles</h1>
      <ul>
        {posts.map((post) => (
          <li key={post.slug}>
            <Link to={`/blog/${post.slug}`}>{post.title}</Link>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default BlogIndexPage; 