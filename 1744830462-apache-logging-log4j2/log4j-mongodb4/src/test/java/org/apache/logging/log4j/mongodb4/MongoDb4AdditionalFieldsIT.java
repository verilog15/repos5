/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.logging.log4j.mongodb4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import com.mongodb.client.FindIterable;
import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.test.junit.LoggerContextSource;
import org.apache.logging.log4j.test.junit.UsingStatusListener;
import org.bson.Document;
import org.junit.jupiter.api.Test;

@UsingMongoDb4
@LoggerContextSource("MongoDb4AdditionalFields.xml")
// Print debug status logger output upon failure
@UsingStatusListener
class MongoDb4AdditionalFieldsIT {

    @Test
    void test(final LoggerContext ctx, final MongoClient mongoClient) {
        final Logger logger = ctx.getLogger(MongoDb4AdditionalFieldsIT.class);
        logger.info("Hello log 1");
        logger.info("Hello log 2", new RuntimeException("Hello ex 2"));
        final MongoDatabase database = mongoClient.getDatabase(MongoDb4TestConstants.DATABASE_NAME);
        assertNotNull(database);
        final MongoCollection<Document> collection =
                database.getCollection(getClass().getSimpleName());
        assertNotNull(collection);
        final FindIterable<Document> found = collection.find();
        final Document first = found.first();
        assertNotNull(first, "first");
        assertEquals("Hello log 1", first.getString("message"), first.toJson());
        assertEquals("INFO", first.getString("level"), first.toJson());
        //
        Document list;
        final String envPath = System.getenv("PATH");
        //
        list = first.get("additionalFields", Document.class);
        assertEquals("1", list.getString("A"), first.toJson());
        assertEquals("2", list.getString("B"), first.toJson());
        assertEquals(envPath, list.getString("env1"), first.toJson());
        assertEquals(envPath, list.getString("env2"), first.toJson());
        //
        found.skip(1);
        final Document second = found.first();
        assertNotNull(second);
        assertEquals("Hello log 2", second.getString("message"), second.toJson());
        assertEquals("INFO", second.getString("level"), second.toJson());
        final Document thrown = second.get("thrown", Document.class);
        assertEquals("Hello ex 2", thrown.getString("message"), thrown.toJson());
        //
        list = second.get("additionalFields", Document.class);
        assertEquals("1", list.getString("A"), first.toJson());
        assertEquals("2", list.getString("B"), first.toJson());
        assertEquals(envPath, list.getString("env1"), first.toJson());
        assertEquals(envPath, list.getString("env2"), first.toJson());
    }
}
